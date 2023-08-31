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

/* Expand macro values to string */
#define STR_VALUE(var)  #var
#define STR(var)        STR_VALUE(var)

#define N_SIZE_MAX      1073741824  /* 1GB */
#define N_SIZE_DEFAULT  N_SIZE_MAX
#define N_ITER_DEFAULT  100
#define CUTS_VERSION    "cuts 1.0"
#define CUTS_CONTACT    "https://github.com/jyvet/cuts"

typedef enum TransferType
{
    HTOD,  /* Host memory to Device (GPU)  */
    DTOH,  /* Device (GPU) to Host memory  */
    DTOD   /* Device (GPU) to Device (GPU) */
} TransferType_t;

typedef struct Transfer
{
    cudaEvent_t    start;   /* Start event for timing purpose                */
    cudaEvent_t    stop;    /* Stop event for timing purpose                 */
    int            device;  /* First (or single) device involved in transfer */
    int            device2; /* Second device involved in the transfer        */
    float         *dest;    /* Source buffer (host or GPU memory)            */
    float         *src;     /* Destination buffer (host or GPU memory)       */
    cudaStream_t   stream;  /* CUDA stream dedicated to the transfer         */
    TransferType_t type;    /* Type and direction of the transfer            */
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
}

/**
 * Cleanup the application
 *
 * @param   cuts[inout]  Main application structure
 */
void fini(Cuts_t *cuts)
{
    free(cuts->transfer);
}

int main(int argc, char *argv[])
{
    Cuts_t cuts;

    init(argc, argv, &cuts);

    for (int i = 0; i < cuts.n_transfers; i++)
    {
        Transfer_t *t = &cuts.transfer[i];

        if (t[i].type == DTOD)
            printf("Transfer %d - P2P transfers from device %d to device %d\n",
                   i, t->device2, t->device);
        else
            printf("Transfer %d - Direct transfers with device %d (%s)\n",
                    i, t->device, (t->type) == DTOH ? "Device to Host" : "Host to Device");
    }

    fini(&cuts);

    return 0;
}
