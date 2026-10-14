/* Two-GPU start-up (DSV4_CUDA_PEER_DEVICE). dsv4_cuda_peer_probe answers
 * before anything is initialised: a refused peer gets no streams, buffers or
 * context. A failed dsv4_cuda_init releases every device it touched and clears
 * the runtime's last error, so the engine's single-device retry starts clean
 * (it used to leak the first attempt, and the first decode_state_set after
 * the retry failed with the stale "invalid device ordinal"), and with several
 * devices the line names the one that failed.
 *
 * The backend is included with counting shims around stream, event and
 * device-allocation lifetimes; failures are injected as REAL runtime errors
 * (bad flags, an impossible size) so they set the last error the way a real
 * fault does. Real devices: the peer parts need two GPUs (skipped with one).
 * A build with no kernel image for device 1 (say CUDA_ARCH=sm_89 on a 3090)
 * exercises the probe's kernel-image refusal and its context release; one
 * with no image for device 0 skips the init parts. About one CUDA context
 * per GPU (~0.3 GiB each) plus a few KiB. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#if !defined(COLI_DSV4_NO_TC)
#include <cublasLt.h>
#endif
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif

static int live_streams, live_events, live_allocs;
static int stream_calls, alloc_calls, fail_stream, fail_alloc, fail_peer_enable;
static cudaError_t test_stream_create(cudaStream_t *s,unsigned flags){
    if(++stream_calls==fail_stream)return cudaStreamCreateWithFlags(s,0xdeadu); /* invalid flags */
    cudaError_t e=cudaStreamCreateWithFlags(s,flags);if(e==cudaSuccess)live_streams++;return e;}
static cudaError_t test_stream_destroy(cudaStream_t s){cudaError_t e=cudaStreamDestroy(s);if(e==cudaSuccess)live_streams--;return e;}
static cudaError_t test_event_create(cudaEvent_t *ev,unsigned flags){cudaError_t e=cudaEventCreateWithFlags(ev,flags);if(e==cudaSuccess)live_events++;return e;}
static cudaError_t test_event_destroy(cudaEvent_t ev){cudaError_t e=cudaEventDestroy(ev);if(e==cudaSuccess)live_events--;return e;}
template<class T> static cudaError_t test_malloc(T **p,size_t n){
    if(++alloc_calls==fail_alloc)return cudaMalloc((void**)p,(size_t)1<<62); /* out of memory */
    cudaError_t e=cudaMalloc((void**)p,n);if(e==cudaSuccess)live_allocs++;return e;}
static cudaError_t test_free(void *p){cudaError_t e=cudaFree(p);if(e==cudaSuccess&&p)live_allocs--;return e;}
/* the contexts outlive shutdown, so a real enable answers "already enabled"
 * from the second init on: fail it through another real runtime error */
static cudaError_t test_enable_peer(int peer,unsigned flags){return fail_peer_enable?cudaSetDevice(-1):cudaDeviceEnablePeerAccess(peer,flags);}
#define cudaStreamCreateWithFlags test_stream_create
#define cudaStreamDestroy test_stream_destroy
#define cudaEventCreateWithFlags test_event_create
#define cudaEventDestroy test_event_destroy
#define cudaMalloc test_malloc
#define cudaFree test_free
#define cudaDeviceEnablePeerAccess test_enable_peer
#include "../backend_cuda_dsv4.cu"
#undef cudaStreamCreateWithFlags
#undef cudaStreamDestroy
#undef cudaEventCreateWithFlags
#undef cudaEventDestroy
#undef cudaMalloc
#undef cudaFree
#undef cudaDeviceEnablePeerAccess

static int failures;
#define CHECK(cond,...) do{if(!(cond)){fprintf(stdout,"FAIL %s:%d: ",__FILE__,__LINE__);fprintf(stdout,__VA_ARGS__);fputc('\n',stdout);failures++;}}while(0)

/* the driver's view: does device d have an active primary context? */
static int has_context(int d){
    CUdevice cd;unsigned flags=0;int active=0;
    if(cuInit(0)!=CUDA_SUCCESS||cuDeviceGet(&cd,d)!=CUDA_SUCCESS||cuDevicePrimaryCtxGetState(cd,&flags,&active)!=CUDA_SUCCESS)return -1;
    return active;}
static int kernel_image(int d){
    cudaFuncAttributes fa;int ok=cudaSetDevice(d)==cudaSuccess&&cudaFuncGetAttributes(&fa,(const void*)mv_fp4_grouped<4>)==cudaSuccess;
    cudaGetLastError();return ok;}
/* nothing held, nothing recorded, no stale error */
static void check_released(const char *what){
    static const Dev zero{};
    cudaError_t last=cudaGetLastError();
    CHECK(last==cudaSuccess,"%s: last error left set (%s)",what,cudaGetErrorString(last));
    CHECK(!live_streams&&!live_events&&!live_allocs,"%s: leaked %d streams, %d events, %d allocations",what,live_streams,live_events,live_allocs);
    CHECK(!ng,"%s: %d devices still registered",what,ng);
    for(int i=0;i<16;i++)CHECK(!memcmp(&g[i],&zero,sizeof zero),"%s: device slot %d not zeroed",what,i);
    for(int a=0;a<16;a++)for(int b=0;b<16;b++)CHECK(!g_peer[a][b],"%s: peer record %d->%d kept",what,a,b);}

/* stderr of one failing init, to check which device it names */
#ifndef _WIN32
static int init_says(const int *dev,int n,const char *needle){
    fflush(stderr);int saved=dup(2);FILE *t=tmpfile();if(saved<0||!t)return -1;
    dup2(fileno(t),2);int r=dsv4_cuda_init(dev,n);fflush(stderr);dup2(saved,2);close(saved);
    char text[4096]={0};rewind(t);size_t got=fread(text,1,sizeof text-1,t);fclose(t);text[got]=0;
    fputs(text,stderr);return !r&&strstr(text,needle)?1:0;}
#endif

int main(void){
    int count=0;
    if(cudaGetDeviceCount(&count)!=cudaSuccess||count<1){cudaGetLastError();puts("SKIP: no CUDA device");return 0;}
    char why[320];

    /* 1. refusals decided from the device list alone touch no device */
    CHECK(!dsv4_cuda_peer_probe(0,0,why,sizeof why)&&strstr(why,"primary"),"peer == primary: \"%s\"",why);
    CHECK(!dsv4_cuda_peer_probe(0,count,why,sizeof why)&&strstr(why,"not a visible"),"peer past the device count: \"%s\"",why);
    CHECK(!dsv4_cuda_peer_probe(-1,0,why,sizeof why)&&strstr(why,"not a visible"),"negative primary: \"%s\"",why);
    CHECK(!dsv4_cuda_peer_probe(0,count,NULL,0),"no reason buffer");
    char tiny[8];
    CHECK(!dsv4_cuda_peer_probe(0,0,tiny,sizeof tiny)&&strlen(tiny)==sizeof tiny-1,"reason not truncated to the buffer");
    CHECK(has_context(0)==0,"the probe created a context on device 0");

    /* 2. the peer itself, before anything else runs on either GPU */
    if(count>=2){
        int ab=0,ba=0;cudaDeviceCanAccessPeer(&ab,0,1);cudaDeviceCanAccessPeer(&ba,1,0);
        int arch=dsv4_cuda_backend_arch_ok(1);
        int usable=dsv4_cuda_peer_probe(0,1,why,sizeof why);
        printf("probe 0 -> 1: %s%s\n",usable?"usable":"refused: ",why);
        CHECK(has_context(0)==0,"probing the peer created a context on the primary");
        /* accepted: the kernel-image check made the peer's context (and shows has_context sees one) */
        CHECK(has_context(1)==usable,"peer context %d after the probe answered %d (%s)",has_context(1),usable,why);
        if(!ab||!ba||!arch)CHECK(!usable,"accepted without peer access both ways / a supported arch");
        else{int image=kernel_image(1);   /* only now: it creates a context on device 1 */
            CHECK(usable==image,"probe %d but kernel image %d for device 1 (%s)",usable,image,why);
            if(!image)CHECK(strstr(why,"kernel image")||strstr(why,"invalid device function"),"refusal does not say why: \"%s\"",why);}
        cudaGetLastError();
    }else puts("SKIP: one GPU, no peer probe");

    if(!kernel_image(0)){printf("SKIP: this build has no kernel image for device 0; init parts not run\n");return failures?1:0;}

    /* 3. a failed init leaves nothing behind and no stale error */
    int bad_second[2]={0,count};
    CHECK(!dsv4_cuda_init(bad_second,2),"init with an invalid second device succeeded");
    check_released("invalid second device");
#ifndef _WIN32
    char needle[64];snprintf(needle,sizeof needle,"device %d did not initialise",count);
    CHECK(init_says(bad_second,2,needle)==1,"failure line does not name device %d",count);
    check_released("invalid second device, again");
#endif
    int pair[2]={0,1},one[1]={0};
    int *ids=count>=2?pair:one,n=count>=2?2:1;
    stream_calls=0;CHECK(dsv4_cuda_init(ids,n),"baseline init failed");
    int streams=stream_calls;dsv4_cuda_shutdown();check_released("baseline shutdown");
    alloc_calls=0;CHECK(dsv4_cuda_init(ids,n),"baseline init failed");
    int allocs=alloc_calls;dsv4_cuda_shutdown();check_released("baseline shutdown");
    for(fail_stream=1;fail_stream<=streams;fail_stream++){
        stream_calls=0;char what[64];snprintf(what,sizeof what,"stream %d of %d failed",fail_stream,streams);
        CHECK(!dsv4_cuda_init(ids,n),"%s: init succeeded",what);check_released(what);}
    fail_stream=0;
    for(fail_alloc=1;fail_alloc<=allocs;fail_alloc++){
        alloc_calls=0;char what[64];snprintf(what,sizeof what,"allocation %d of %d failed",fail_alloc,allocs);
        CHECK(!dsv4_cuda_init(ids,n),"%s: init succeeded",what);check_released(what);}
    fail_alloc=0;
#ifndef _WIN32
    if(n==2){stream_calls=0;fail_stream=1;   /* device 0's first stream */
        CHECK(init_says(ids,2,"device 0 did not initialise")==1,"failure line does not name device 0");
        fail_stream=0;check_released("device 0 failed");}
#endif
    if(n==2){int ab=0,ba=0;cudaDeviceCanAccessPeer(&ab,0,1);cudaDeviceCanAccessPeer(&ba,1,0);cudaGetLastError();
        if(ab||ba){fail_peer_enable=1;CHECK(!dsv4_cuda_init(ids,2),"init succeeded with peer access failing");
            fail_peer_enable=0;check_released("enable peer access failed");}}

    /* 4. the engine's retry: one device, from the state a failed init left */
    CHECK(!dsv4_cuda_init(bad_second,2),"init with an invalid second device succeeded");
    CHECK(dsv4_cuda_init(one,1),"single-device retry failed");
    CHECK(dsv4_cuda_decode_state_set(0,1,2),"first decode_state_set after the retry failed");
    CHECK(cudaDeviceSynchronize()==cudaSuccess,"device 0 faulted");
    dsv4_cuda_shutdown();check_released("retry shutdown");

    /* 5. a usable peer: init records access both ways, shutdown forgets it */
    if(count>=2&&dsv4_cuda_peer_probe(0,1,why,sizeof why)){
        CHECK(dsv4_cuda_init(pair,2),"two-device init failed");
        CHECK(dsv4_cuda_peer_access(0,1)&&dsv4_cuda_peer_access(1,0),"peer access not recorded");
        dsv4_cuda_shutdown();
        CHECK(!dsv4_cuda_peer_access(0,1),"peer access outlived shutdown");
        check_released("two-device shutdown");
    }
    if(failures){printf("dsv4 peer init: %d FAILED\n",failures);return 1;}
    printf("dsv4 peer init: probe, init rollback and retry: OK (%d device%s)\n",count,count>1?"s":"");
    return 0;
}
