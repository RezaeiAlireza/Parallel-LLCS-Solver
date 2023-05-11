#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define LEN 51200
#define PERTASK_V1 1024
#define PERTASK_V2 512
#define PERTASK_V3 256
#define TSIZE 512
#define MIN(x, y) (((x) <= (y)) ? (x) : (y))
#define MAX(x, y) (((x) >= (y)) ? (x) : (y))

unsigned long long llcs_serial(const char *X, const char *Y, unsigned int **M)
{
    unsigned long long entries_visited = 0;

    int i = 0;
    while (i < LEN)
    {
        int j = 0;
        while (j < LEN)
        {
            if (X[i] == Y[j]) {
                M[i + 1][j + 1] = M[i][j] + 1;
            } else if (M[i + 1][j] < M[i][j + 1]) {
                M[i + 1][j + 1] = M[i][j + 1];
            } else {
                M[i + 1][j + 1] = M[i + 1][j];
            }

            entries_visited++;

            j++;
        }
        i++;
    }
    return entries_visited;
}


#if defined(_OPENMP)

unsigned long long llcs_parallel_tasks(const char *X, const char *Y, unsigned int **M)
{
    unsigned long long entries_visited = 0;
    #pragma omp parallel
    #pragma omp single
    {    
        int adLine;
        for(adLine = 1; adLine <= LEN+LEN-1; adLine+=PERTASK_V1){
            int rowN, colN;               
            for(rowN =  MIN(LEN-PERTASK_V1+1, adLine);rowN >= MAX(1, adLine-LEN+1);rowN-=PERTASK_V1)
            {
                colN = adLine - rowN + 1;
                int localVisit = 0;
                #pragma omp task default(none) \
                        shared( X, Y, M, entries_visited, adLine) \
                            firstprivate(localVisit, rowN, colN) \
                                depend(in:M[rowN+PERTASK_V1-1][colN-1]) \
                                    depend(in:M[rowN-1][colN+PERTASK_V1-1]) \
                                        depend(out:M[rowN+PERTASK_V1-1][colN+PERTASK_V1-1])
                {
                    for (int i = rowN;i < rowN + PERTASK_V1; i++)
                    {
                        for (int j = colN; j < colN + PERTASK_V1; j++)
                        {
                            if (X[i-1] == Y[j-1]) {
                                M[i][j] = M[i-1][j-1] + 1;
                            } else if (M[i][j-1] < M[i-1][j]) {
                                M[i][j] = M[i-1][j];
                            } else {
                                M[i][j] = M[i][j-1];
                            }
                            localVisit++;
                        }
                    }
                    #pragma omp atomic
                    entries_visited += localVisit;
                }
            }
        }
    }
    return entries_visited;
}
unsigned long long llcs_parallel_tasks_v2(const char *X, const char *Y, unsigned int **M)
{
    unsigned long long entries_visited = 0;
    #pragma omp parallel
    #pragma omp single
    {    
        int adLine;
        for(adLine = 1; adLine <= LEN+LEN-1; adLine+=PERTASK_V2){
            int rowN, colN;               
            for(rowN =  MIN(LEN-PERTASK_V2+1, adLine);rowN >= MAX(1, adLine-LEN+1);rowN-=PERTASK_V2)
            {
                colN = adLine - rowN + 1;
                int localVisit = 0;
                #pragma omp task default(none) \
                        shared( X, Y, M, entries_visited, adLine) \
                            firstprivate(localVisit, rowN, colN) \
                                depend(in:M[rowN+PERTASK_V2-1][colN-1]) \
                                    depend(in:M[rowN-1][colN+PERTASK_V2-1]) \
                                        depend(out:M[rowN+PERTASK_V2-1][colN+PERTASK_V2-1])
                {
                    for (int i = rowN;i < rowN + PERTASK_V2; i++)
                    {
                        for (int j = colN; j < colN + PERTASK_V2; j++)
                        {
                            if (X[i-1] == Y[j-1]) {
                                M[i][j] = M[i-1][j-1] + 1;
                            } else if (M[i][j-1] < M[i-1][j]) {
                                M[i][j] = M[i-1][j];
                            } else {
                                M[i][j] = M[i][j-1];
                            }
                            localVisit++;
                        }
                    }
                    #pragma omp atomic
                    entries_visited += localVisit;
                }
            }
        }
    }
    return entries_visited;
}
    
unsigned long long llcs_parallel_tasks_v3(const char *X, const char *Y, unsigned int **M)
{
    unsigned long long entries_visited = 0;
    #pragma omp parallel
    #pragma omp single
    {    
        int adLine;
        for(adLine = 1; adLine <= LEN+LEN-1; adLine+=PERTASK_V3){
            int rowN, colN;               
            for(rowN =  MIN(LEN-PERTASK_V3+1, adLine);rowN >= MAX(1, adLine-LEN+1);rowN-=PERTASK_V3)
            {
                colN = adLine - rowN + 1;
                int localVisit = 0;
                #pragma omp task default(none) \
                        shared( X, Y, M, entries_visited, adLine) \
                            firstprivate(localVisit, rowN, colN) \
                                depend(in:M[rowN+PERTASK_V3-1][colN-1]) \
                                    depend(in:M[rowN-1][colN+PERTASK_V3-1]) \
                                        depend(out:M[rowN+PERTASK_V3-1][colN+PERTASK_V3-1])
                {
                    for (int i = rowN;i < rowN + PERTASK_V3; i++)
                    {
                        for (int j = colN; j < colN + PERTASK_V3; j++)
                        {
                            if (X[i-1] == Y[j-1]) {
                                M[i][j] = M[i-1][j-1] + 1;
                            } else if (M[i][j-1] < M[i-1][j]) {
                                M[i][j] = M[i-1][j];
                            } else {
                                M[i][j] = M[i][j-1];
                            }
                            localVisit++;
                        }
                    }
                    #pragma omp atomic
                    entries_visited += localVisit;
                }
            }
        }
    }
    return entries_visited;
}

unsigned long long llcs_parallel_taskloop(const char *X, const char *Y, unsigned int **M)
{
    unsigned long long entries_visited = 0;
    #pragma omp parallel   
    #pragma omp single
    {
        int rowN, colN, adLine;
        for(adLine = 1; adLine <= LEN+LEN-1; adLine+=TSIZE)
        {
            #pragma omp taskloop reduction(+:entries_visited) \
                    grainsize(1)
            for(rowN =  MIN(LEN-TSIZE+1, adLine); rowN >= MAX(1, adLine-LEN+1); rowN-=TSIZE)
            {
                colN = adLine - rowN + 1;
                for (int i = rowN;i < rowN + TSIZE; i++)
                {
                    for (int j = colN; j < colN + TSIZE; j++)
                    {
                        if (X[i-1] == Y[j-1]) {
                            M[i][j] = M[i-1][j-1] + 1;
                        } else if (M[i][j-1] < M[i-1][j]) {
                            M[i][j] = M[i-1][j];
                        } else {
                            M[i][j] = M[i][j-1];
                        }
                        entries_visited++;
                    }
                }
            }
        }   
    }
    return entries_visited;
}
unsigned long long llcs_parallel_taskloop_v2(const char *X, const char *Y, unsigned int **M)
{
    unsigned long long entries_visited = 0;
    #pragma omp parallel   
    #pragma omp single
    {
        int rowN, colN, adLine;
        for(adLine = 1; adLine <= LEN+LEN-1; adLine+=TSIZE)
        {
            #pragma omp taskloop reduction(+:entries_visited) \
                    grainsize(2)
            for(rowN =  MIN(LEN-TSIZE+1, adLine); rowN >= MAX(1, adLine-LEN+1); rowN-=TSIZE)
            {
                colN = adLine - rowN + 1;
                for (int i = rowN;i < rowN + TSIZE; i++)
                {
                    for (int j = colN; j < colN + TSIZE; j++)
                    {
                        if (X[i-1] == Y[j-1]) {
                            M[i][j] = M[i-1][j-1] + 1;
                        } else if (M[i][j-1] < M[i-1][j]) {
                            M[i][j] = M[i-1][j];
                        } else {
                            M[i][j] = M[i][j-1];
                        }
                        entries_visited++;
                    }
                }
            }
        }   
    }
    return entries_visited;
}
unsigned long long llcs_parallel_taskloop_v3(const char *X, const char *Y, unsigned int **M)
{
    unsigned long long entries_visited = 0;
    #pragma omp parallel   
    #pragma omp single
    {
        int rowN, colN, adLine;
        for(adLine = 1; adLine <= LEN+LEN-1; adLine+=TSIZE)
        {
            #pragma omp taskloop reduction(+:entries_visited) \
                    grainsize(3)
            for(rowN =  MIN(LEN-TSIZE+1, adLine); rowN >= MAX(1, adLine-LEN+1); rowN-=TSIZE)
            {
                colN = adLine - rowN + 1;
                for (int i = rowN;i < rowN + TSIZE; i++)
                {
                    for (int j = colN; j < colN + TSIZE; j++)
                    {
                        if (X[i-1] == Y[j-1]) {
                            M[i][j] = M[i-1][j-1] + 1;
                        } else if (M[i][j-1] < M[i-1][j]) {
                            M[i][j] = M[i-1][j];
                        } else {
                            M[i][j] = M[i][j-1];
                        }
                        entries_visited++;
                    }
                }
            }
        }   
    }
    return entries_visited;
}

#endif