/**
* CUDA Transfer Streams (CUts): Application designed to launch intra-node
*                               transfer streams in an adjustable way.
* URL       https://github.com/jyvet/cuts
* License   MIT
* Author    Jean-Yves VET <contact[at]jean-yves.vet>
* Copyright (c) 2023
******************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <argp.h>
#include <unistd.h>
#include <cuda.h>
#include <pthread.h>
#include <numa.h>

/* Expand macro values to string */
#define STR_VALUE(var)  #var
#define STR(var)        STR_VALUE(var)

#define N_SIZE_MAX      1073741824  /* 1GiB */
#define N_SIZE_DEFAULT  N_SIZE_MAX
#define N_ITER_DEFAULT  100
#define CUTS_VERSION    "cuts 1.0"
#define CUTS_CONTACT    "https://github.com/jyvet/cuts"

#define checkCuda(ret) { assertCuda((ret), __FILE__, __LINE__); }
inline void assertCuda(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CheckCuda: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

typedef enum TransferType
{
    HTOD = 0,  /* Host memory to Device (GPU)  */
    DTOH,      /* Device (GPU) to Host memory  */
    DTOD,      /* Device (GPU) to Device (GPU) */
} TransferType_t;

const char * const ttype_str[] =
{
    "Host to Device",
    "Device to Host",
    "Device to Device",
};

typedef struct Transfer
{
    cudaEvent_t    start;     /* Start event for timing purpose                */
    cudaEvent_t    stop;      /* Stop event for timing purpose                 */
    int            device;    /* First (or single) device involved in transfer */
    int            device2;   /* Second device involved in the transfer        */
    float         *dest;      /* Source buffer (host or GPU memory)            */
    float         *src;       /* Destination buffer (host or GPU memory)       */
    cudaStream_t   stream;    /* CUDA stream dedicated to the transfer         */
    TransferType_t type;      /* Type and direction of the transfer            */
    int            numa_node; /* NUMA node locality                            */
    struct cudaDeviceProp prop_device;
    struct cudaDeviceProp prop_device2;
} Transfer_t;

typedef struct Cuts
{
    Transfer_t *transfer;      /* Array containing all transfers to launch */
    int         n_transfers;   /* Amount of transfers                      */
    long        n_iter;        /* Amount of iterations for each transfer   */
    long        n_size;        /* Transfer size in bytes                   */
    bool        is_numa_aware; /* Allocate the buffers on the proper NUMA node */
} Cuts_t;

const char *argp_program_version = CUTS_VERSION;
const char *argp_program_bug_address = CUTS_CONTACT;

/* Program documentation */
static char doc[] = "This application is designed to launch intra-node transfer streams "
                    "in an adjustable way. It may trigger different types of CUDA transfers "
                    "concurrently. Each transfer is bound to a CUDA stream. Transfer "
                    "buffers in main memory are allocated (by default) on the proper "
                    "NUMA node. The application accepts the following arguments:";

/* A description of the arguments we accept (in addition to the options) */
static char args_doc[] = "--dtoh=<gpu_id> --htod=<gpu_id> --dtod=<dest_gpu_id,src_gpu_id>";

/* Options */
static struct argp_option options[] =
{
    {"dtoh",             'd', "<id>",    0,  "Provide GPU id for Device to Host transfer."},
    {"htod",             'h', "<id>",    0,  "Provide GPU id for Host to Device transfer."},
    {"dtod",             'p', "<id,id>", 0,  "Provide comma-separated GPU ids to specify which "
                                             "pair of GPUs to use for peer to peer transfer. "
                                             "Firt id is the destination, second id is the source."},
    {"iter",             'i', "<nb>",    0,  "Specify the amount of iterations. [default: "
                                              STR(N_ITER_DEFAULT) "]"},
    {"no-numa-affinity", 'n', 0,         0,  "Do not make the transfer buffers NUMA aware."},
    {"size",             's', "<bytes>", 0,  "Specify the transfer size in bytes. [default: "
                                              STR(N_SIZE_DEFAULT) "]"},
    {0}
};

/* Parse a single option */
static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    Cuts_t *cuts = (Cuts_t *)state->input;
    Transfer_t *transfer = &cuts->transfer[cuts->n_transfers];

    const char* token;
    char *endptr;

    switch (key)
    {
        case 'd':
            transfer->type = DTOH;
            transfer->device = strtol(arg, &endptr, 10);
            if (errno == EINVAL || errno == ERANGE || transfer->device < 0)
            {
                fprintf(stderr, "Error: cannot parse the GPU id from the --dtoh argument. "
                                "Exit.\n");
                exit(1);
            }

            transfer->device2 = -1;
            cuts->n_transfers++;
            break;
        case 'h':
            transfer->type = HTOD;
            transfer->device = strtol(arg, &endptr, 10);
            if (errno == EINVAL || errno == ERANGE || transfer->device < 0)
            {
                fprintf(stderr, "Error: cannot parse the GPU id from th --htod argument. "
                                "Exit.\n");
                exit(1);
            }

            transfer->device2 = -1;
            cuts->n_transfers++;
            break;
        case 'i':
            cuts->n_iter = strtol(arg, &endptr, 10);
            if (errno == EINVAL || errno == ERANGE || cuts->n_iter < 0)
            {
                fprintf(stderr, "Error: cannot parse the amount of iterations from the "
                                "--iter argument. Exit.\n");
                exit(1);
            }
            break;
        case 'n':
            cuts->is_numa_aware = false;
            break;
        case 'p':
            transfer->type = DTOD;

            /* Parse first GPU id */
            token = strtok(arg, ",");
            transfer->device = (token != NULL ) ? strtol(token, &endptr, 10) : -1;
            if (errno == EINVAL || errno == ERANGE || token == endptr || transfer->device < 0)
            {
                fprintf(stderr, "Error: cannot parse first GPU id from --dtod argument. "
                                "This argument only accepts a list of two ids separated "
                                "by a comma. Exit.\n");
                exit(1);
            }

            /* Parse second GPU id */
            token = strtok(NULL, ",");
            transfer->device2 = (token != NULL) ? strtol(token, &endptr, 10) : -1;
            if (errno == EINVAL || errno == ERANGE || token == endptr || transfer->device2 < 0)
            {
                fprintf(stderr, "Error: cannot parse second GPU id from --dtod argument. "
                                "This argument only accepts a list of two ids separated "
                                "by a comma. Exit.\n");
                exit(1);
            }

            /* Ensure there is no further ids */
            token = strtok(NULL, ",");
            if (token != endptr && token != NULL)
            {
                fprintf(stderr, "Error: --dtod argument only accepts a list of two GPU ids "
                                "separated by a comma. Exit.\n");
                exit(1);
            }

            cuts->n_transfers++;
            break;
        case 's':
            cuts->n_size = strtol(arg, &endptr, 10);
            if (errno == EINVAL || errno == ERANGE || cuts->n_iter < 0)
            {
                fprintf(stderr, "Error: cannot parse the transfer size from the "
                                "--size argument. Exit.\n");
                exit(1);
            }

            if (cuts->n_size > N_SIZE_MAX)
            {
                fprintf(stderr, "Error: maximum transfer size value is %d. Exit.", N_SIZE_MAX);
                exit(1);
            }
            break;
        case ARGP_KEY_END:
            if (cuts->n_transfers == 0)
                argp_usage(state);
            break;
        default:
            return ARGP_ERR_UNKNOWN;
    }

    return 0;
}

/* Argp parser */
static struct argp argp = { options, parse_opt, args_doc, doc };

static void _transfer_init_common(Transfer_t *t)
{
    t->numa_node = -1;

    checkCuda( cudaGetDeviceProperties(&t->prop_device, t->device) );
    if (t->device2 >= 0)
        checkCuda( cudaGetDeviceProperties(&t->prop_device2, t->device2) );

    checkCuda( cudaSetDevice(t->device) );

    checkCuda( cudaEventCreate(&t->start) );
    checkCuda( cudaEventCreate(&t->stop) );

    checkCuda( cudaStreamCreateWithFlags(&t->stream, cudaStreamNonBlocking) );
}

/**
 * Set NUMA affinity based on GPU property.
 *
 * @param   t[in]  transfer structure
 */
void set_numa_affinity(Transfer_t *t)
{
    char numa_file[PATH_MAX];
    struct cudaDeviceProp *prop = &t->prop_device;
    sprintf(numa_file, "/sys/class/pci_bus/0000:%.2x/device/numa_node", prop->pciBusID);

    FILE* file = fopen(numa_file, "r");
    if (file == NULL)
        return;

    int ret = fscanf(file, "%d", &t->numa_node);
    fclose(file);

    if (ret == 1)
        numa_set_preferred(t->numa_node);
}

void dtoh_transfer_init(Transfer_t *t, const size_t n_bytes, const bool is_numa_aware)
{
    _transfer_init_common(t);

    if (is_numa_aware)
        set_numa_affinity(t);

    checkCuda( cudaMalloc(((void **)&t->src), n_bytes) );
    checkCuda( cudaHostAlloc(((void **)&t->dest), n_bytes, cudaHostAllocDefault) );
}

void htod_transfer_init(Transfer_t *t, const size_t n_bytes, const bool is_numa_aware)
{
    _transfer_init_common(t);

    if (is_numa_aware)
        set_numa_affinity(t);

    checkCuda( cudaMalloc(((void **)&t->dest), n_bytes) );
    checkCuda( cudaHostAlloc(((void **)&t->src), n_bytes, cudaHostAllocDefault) );
}

void dtod_transfer_init(Transfer_t *t, const size_t n_bytes)
{
    _transfer_init_common(t);

    /* Ensure peer-to-peer access is possible between the two GPUs */
    int is_access = 0;
    cudaDeviceCanAccessPeer(&is_access, t->device, t->device2);
    if (!is_access)
    {
        fprintf(stderr, "Error: P2P cannot be enabled between devices %d and %d\n",
                t->device, t->device2);
        exit(1);
    }

    checkCuda( cudaSetDevice(t->device) );
    checkCuda( cudaMalloc((void **)&t->dest, n_bytes) );
    checkCuda( cudaDeviceEnablePeerAccess(t->device2, 0) );

    checkCuda( cudaSetDevice(t->device2) );
    checkCuda( cudaMalloc((void **)&t->src, n_bytes) );
}

/**
 * Initialize all transfers
 *
 * @param   cuts[inout]  Main application structure
 */
void transfer_init(Cuts_t *cuts)
{
    /* Initialize all streams and buffers */
    for (int i = 0; i < cuts->n_transfers; i++)
    {
        Transfer_t *t = &cuts->transfer[i];

        switch(t->type)
        {
            case DTOH:
                dtoh_transfer_init(t, cuts->n_size, cuts->is_numa_aware);
                break;
            case HTOD:
                htod_transfer_init(t, cuts->n_size, cuts->is_numa_aware);
                break;
            case DTOD:
                dtod_transfer_init(t, cuts->n_size);
                break;
        }
    }
}

/**
 * Initialize the application
 *
 * @param   argc[in]    Amount of arguments
 * @param   argv[in]    Array of arguments
 * @param   cuts[out]   Main application structure
 */
void init(int argc, char *argv[], Cuts_t *cuts)
{
    cuts->transfer = (Transfer_t *)malloc(sizeof(Transfer_t) * (argc - 1));
    if (cuts->transfer == NULL)
    {
         fprintf(stderr,"Error: Cannot allocate main data structure. Exit.\n");
         exit(1);
    }

    /* Set defaults */
    cuts->n_transfers   = 0;
    cuts->n_iter        = N_ITER_DEFAULT;
    cuts->n_size        = N_SIZE_DEFAULT;
    cuts->is_numa_aware = true;

    argp_parse(&argp, argc, argv, 0, 0, cuts);

    transfer_init(cuts);
}

/**
 * Cleanup the application
 *
 * @param   cuts[inout]  Main application structure
 */
void fini(Cuts_t *cuts)
{
    /* Free host buffers */
    for (int i = 0; i < cuts->n_transfers; i++)
    {
        Transfer_t *t = &cuts->transfer[i];

        switch(t->type)
        {
            case DTOH:
                checkCuda( cudaFreeHost(t->dest) );
                break;
            case HTOD:
                checkCuda( cudaFreeHost(t->src) );
                break;
        }
    }

    free(cuts->transfer);
}

/**
 * Launch a direct transfer stream (Host to Device or Device to Host)
 *
 * @param   t[inout]     Transfe data
 * @param   n_bytes[in]  Transfer size
 * @param   n_iter[in]   Iterations
 */
void direct_transfer(Transfer_t *t, const size_t n_bytes, const size_t n_iter)
{
    checkCuda( cudaSetDevice(t->device) );

    printf("Launching %s transfers with Device %d (0x%.2x)",
           ttype_str[t->type], t->device, t->prop_device.pciBusID);
    if (t->numa_node >= 0)
        printf(" - Host buffer allocated on NUMA node %d", t->numa_node);

    printf("\n");

    checkCuda( cudaEventRecord(t->start, t->stream) );

    for (size_t i = 0; i < n_iter; i++)
    {
        checkCuda( cudaMemcpyAsync(t->dest, t->src, n_bytes, (t->type == DTOH) ?
                                   cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, t->stream) );
    }

    checkCuda( cudaEventRecord(t->stop, t->stream) );
}

/**
 * Launch a peer-to-peer transfer stream
 *
 * @param   t[inout]     Transfe data
 * @param   n_bytes[in]  Transfer size
 * @param   n_iter[in]   Iterations
 */
void dtod_transfer(Transfer_t *t, const size_t n_bytes, const size_t n_iter)
{
    checkCuda( cudaSetDevice(t->device) );

    printf("Launching P2P PCIe transfers from Device %d (0x%.2x) to Device %d (0x%.2x)\n",
           t->device2, t->prop_device2.pciBusID, t->device, t->prop_device.pciBusID);

    checkCuda( cudaEventRecord(t->start, t->stream) );

    for (size_t i = 0; i < n_iter; i++)
    {
        checkCuda( cudaMemcpyPeerAsync(t->dest, t->device, t->src, t->device2, n_bytes, t->stream) );
    }

    checkCuda( cudaEventRecord(t->stop, t->stream) );
}

/**
 * Display a dot every second as Heartbeat. Stop when transfers are completed.
 *
 * @param   arg[in]  Pointer to transfer state
 */
void* heart_beat(void *arg)
{
    bool *is_transfering = (bool*)arg;
    setbuf(stdout, NULL);

    while (*is_transfering)
    {
        printf(".");
        sleep(1);
    }

    return NULL;
}

int main(int argc, char *argv[])
{
    Cuts_t cuts;

    init(argc, argv, &cuts);
    const int n_transfers = cuts.n_transfers;
    const size_t n_iter = cuts.n_iter;
    const size_t n_bytes = cuts.n_size;
    const float n_gbytes = (float)n_bytes / 1E9;
    bool is_transfering = true;
    pthread_t thread;

    /* Start all transfers at the same time */
    for (int i = 0; i < n_transfers; i++)
    {
        Transfer_t *t = &cuts.transfer[i];
        (t->type == DTOD) ? dtod_transfer(t, n_bytes, n_iter) : direct_transfer(t, n_bytes, n_iter);
    }

    /* Starting heartbeat thread */
    pthread_create(&thread, NULL, &heart_beat, &is_transfering);

    /* Synchronize the GPU from each transfer */
    for (int i = 0; i < n_transfers; i++)
    {
        Transfer_t *t = &cuts.transfer[i];
        checkCuda( cudaSetDevice(t->device) );
        checkCuda( cudaDeviceSynchronize() );
    }

    is_transfering = false;
    printf("\nCompleted.\n");

    /* Print bandwidth results */
    for (int i = 0; i < n_transfers; i++)
    {
        float dt_msec, dt_sec;
        Transfer_t *t = &cuts.transfer[i];
        checkCuda( cudaSetDevice(t->device) );
        checkCuda( cudaEventElapsedTime(&dt_msec, t->start, t->stop) );
        dt_sec = dt_msec / 1E3;

        if (t->type == DTOD)
            printf("Transfer %d - P2P transfers from Device %d (0x%.2x) to Device %d (0x%.2x):"
                   " %.3f GB/s  (%.2f seconds)\n", i, t->device2, t->prop_device2.pciBusID,
                   t->device, t->prop_device.pciBusID, n_gbytes / dt_sec * n_iter, dt_sec);
        else
            printf("Transfer %d - Direct transfers (%s) with Device %d (0x%.2x): "
                   "%.3f GB/s  (%.2f seconds)\n", i, ttype_str[t->type],
                   t->device, t->prop_device.pciBusID, n_gbytes / dt_sec * n_iter, dt_sec);
    }

    fini(&cuts);

    return 0;
}
