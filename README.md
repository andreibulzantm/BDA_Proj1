# BDA_Proj1

Codul nostru e în unified.py. Am pus și două versiuni găsite pe internet pentru cmeans/kmeans separat, nu știu dacă vă ajută la ceva sau nu.
Am pus în comentarii în cod ca să se vadă unde se găsesc fiecare din pașii ăștia din explicație:

The steps of UF algorithm are UF1 to UF6:
Step UF1. Initialize the iteration index =1 and the number k of clusters.
Step UF2. Initialize the centroid vector c(1) (the initial value of the solution to (21)) expressed in (9).
Step UF3. Compute the distances   for each combination of records xi, i=1...n, and centroids cj, j=1...k, using (6).
Step UF4. Build the membership matrix  
.....
where the elements of the matrix,  , i=1...n, j=1...k, are computed using (22).
Step UF5. Compute the new centroids   used in the next iteration in terms of the following relationship:
.....
where j=1...k, and express the centroid vector c(+1) used in the next iteration and defined in (12).
Step UF6. This step is identical to step FCM6 of FCM algorithm. If one or both stopping criteria expressed in (13) and (15) is/are fulfilled, the algorithm is complete and the solution to (21) is ...
... (30)
Otherwise, a new iteration will be performed, i.e.,  is replaced with +1 and the algorithm will continue with step UF2.
The above UF algorithm implementation results directly in FCM algorithm implementation or KM algorithm implementation by only particularizing (22).

Pașii ăștia sunt în fișierul cu Partitional implementation.
O să șterg fișierele de input + kmeans, cmeans după ce vă uitați și vedeți ce vă ajută și că merge.
