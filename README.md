# Algoritmo di Strassen in ambiente GPU-CUDA
Il progetto sviluppato propone una strategia per l'accelerazione del prodotto tra due matrici in ambiente GPGPU. L'algoritmo presente ricorre all'algoritmo di Strassen per ridurre il tempo computazionale dell'algoritmo, studiandone anche le peculiarità.
La strategia di parallelizzazione è delegata alla libreria cuBLAS, che svolge le operazioni di addizione e sottrazione matriciale.

Durante la progettazione, è stata ideata in un primo momento un'implementazione in ambiente CPU, per poi "trasporre" la logica in ambiente GPGPU, cercando di mantenere quanto più uniforme possibile i due codici.

## Definizione algoritmo di Strassen
Siano date due matrici quadrate <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\color{White}\mathbf{A}">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\mathbf{A}">
  <img src="https://latex.codecogs.com/svg.image?\mathbf{A}">
</picture> e <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\color{White}\mathbf{B}">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\mathbf{B}">
  <img src="https://latex.codecogs.com/svg.image?\mathbf{B}">
</picture> di dimensioni <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\color{White}n\times&space;n">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?n\times&space;n">
  <img src="https://latex.codecogs.com/svg.image?n\times&space;n">
</picture>. Si vuole calcolare la matrice <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\color{White}\mathbf{C}=\mathbf{A}\times\mathbf{B}">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\mathbf{C}=\mathbf{A}\times\mathbf{B}">
  <img src="https://latex.codecogs.com/svg.image?\mathbf{C}=\mathbf{A}\times\mathbf{B}">
</picture> con l'algoritmo di Strassen.
L’algoritmo ricava la matrice risultato definendo sette nuove matrici

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\large&space;\color{White}\mathbf{M}_1=(\mathbf{A}_{11}&plus;\mathbf{A}_{22})(\mathbf{B}_{11}&plus;\mathbf{B}_{22})">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\large&space;\mathbf{M}_1=(\mathbf{A}_{11}&plus;\mathbf{A}_{22})(\mathbf{B}_{11}&plus;\mathbf{B}_{22})">
  <img src="https://latex.codecogs.com/svg.image?\large&space;\mathbf{M}_1=(\mathbf{A}_{11}&plus;\mathbf{A}_{22})(\mathbf{B}_{11}&plus;\mathbf{B}_{22})">
</picture>
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\large&space;\color{White}\mathbf{M}_2=(\mathbf{A}_{21}&plus;\mathbf{A}_{22})\mathbf{B}_{11}">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\large&space;\mathbf{M}_2=(\mathbf{A}_{21}&plus;\mathbf{A}_{22})\mathbf{B}_{11}">
  <img src="https://latex.codecogs.com/svg.image?\large&space;\mathbf{M}_2=(\mathbf{A}_{21}&plus;\mathbf{A}_{22})\mathbf{B}_{11}">
</picture>
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\large&space;\color{White}\mathbf{M}_3=\mathbf{A}_{11}(\mathbf{B}_{12}-\mathbf{B}_{22})">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\large&space;\mathbf{M}_3=\mathbf{A}_{11}(\mathbf{B}_{12}-\mathbf{B}_{22})">
  <img src="https://latex.codecogs.com/svg.image?\large&space;\mathbf{M}_3=\mathbf{A}_{11}(\mathbf{B}_{12}-\mathbf{B}_{22})">
</picture>
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\large&space;\color{White}\mathbf{M}_4=\mathbf{A}_{22}(\mathbf{B}_{21}-\mathbf{B}_{11})">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\large&space;\mathbf{M}_4=\mathbf{A}_{22}(\mathbf{B}_{21}-\mathbf{B}_{11})">
  <img src="https://latex.codecogs.com/svg.image?\large&space;\mathbf{M}_4=\mathbf{A}_{22}(\mathbf{B}_{21}-\mathbf{B}_{11})">
</picture>
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\large&space;\color{White}\mathbf{M}_5=(\mathbf{A}_{11}&plus;\mathbf{A}_{12})\mathbf{B}_{22}">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\large&space;\mathbf{M}_5=(\mathbf{A}_{11}&plus;\mathbf{A}_{12})\mathbf{B}_{22}">
  <img src="https://latex.codecogs.com/svg.image?\large&space;\mathbf{M}_5=(\mathbf{A}_{11}&plus;\mathbf{A}_{12})\mathbf{B}_{22}">
</picture>
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\large&space;\color{White}\mathbf{M}_6=(\mathbf{A}_{21}-\mathbf{A}_{11})(\mathbf{B}_{11}&plus;\mathbf{B}_{12})">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\large&space;\mathbf{M}_6=(\mathbf{A}_{21}-\mathbf{A}_{11})(\mathbf{B}_{11}&plus;\mathbf{B}_{12})">
  <img src="https://latex.codecogs.com/svg.image?\large&space;\mathbf{M}_6=(\mathbf{A}_{21}-\mathbf{A}_{11})(\mathbf{B}_{11}&plus;\mathbf{B}_{12})">
</picture>
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\large&space;\color{White}\mathbf{M}_7=(\mathbf{A}_{12}-\mathbf{A}_{22})(\mathbf{B}_{21}&plus;\mathbf{B}_{22})">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\large&space;\mathbf{M}_7=(\mathbf{A}_{12}-\mathbf{A}_{22})(\mathbf{B}_{21}&plus;\mathbf{B}_{22})">
  <img src="https://latex.codecogs.com/svg.image?\large&space;\mathbf{M}_7=(\mathbf{A}_{12}-\mathbf{A}_{22})(\mathbf{B}_{21}&plus;\mathbf{B}_{22})">
</picture>
<br>
che riduce, di fatto, la complessità computazionale dell'algoritmo a <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\color{White}O(n^{\log_27})">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?O(n^{\log_27})">
  <img src="https://latex.codecogs.com/svg.image?O(n^{\log_27})">
</picture> in quanto da 8 prodotti ne vengono computati "solo" 7. Successivamente si ricombina nella matrice risultato

<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\large&space;\color{White}\begin{pmatrix}\mathbf{C}_{11}&\mathbf{C}_{12}\\\mathbf{C}_{21}&\mathbf{C}_{22}\end{pmatrix}=\begin{pmatrix}\mathbf{M}_1&plus;\mathbf{M}_4-\mathbf{M}_5&plus;\mathbf{M}_7&\mathbf{M}_3&plus;\mathbf{M}_5\\\mathbf{M}_2&plus;\mathbf{M}_4&\mathbf{M}_1-\mathbf{M}_2&plus;\mathbf{M}_3&plus;\mathbf{M}_6\end{pmatrix}">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\large&space;\begin{pmatrix}\mathbf{C}_{11}&\mathbf{C}_{12}\\\mathbf{C}_{21}&\mathbf{C}_{22}\end{pmatrix}=\begin{pmatrix}\mathbf{M}_1&plus;\mathbf{M}_4-\mathbf{M}_5&plus;\mathbf{M}_7&\mathbf{M}_3&plus;\mathbf{M}_5\\\mathbf{M}_2&plus;\mathbf{M}_4&\mathbf{M}_1-\mathbf{M}_2&plus;\mathbf{M}_3&plus;\mathbf{M}_6\end{pmatrix}">
  <img src="https://latex.codecogs.com/svg.image?\large&space;\begin{pmatrix}\mathbf{C}_{11}&\mathbf{C}_{12}\\\mathbf{C}_{21}&\mathbf{C}_{22}\end{pmatrix}=\begin{pmatrix}\mathbf{M}_1&plus;\mathbf{M}_4-\mathbf{M}_5&plus;\mathbf{M}_7&\mathbf{M}_3&plus;\mathbf{M}_5\\\mathbf{M}_2&plus;\mathbf{M}_4&\mathbf{M}_1-\mathbf{M}_2&plus;\mathbf{M}_3&plus;\mathbf{M}_6\end{pmatrix}">
</picture>


Il "trucco" dell'algoritmo consiste nell'evitare il prodotto riga per colonna (algoritmo standard del prodotto matriciale) matrici, andando a ridurre il problema fintantoché le sette matrici non siano ricavabili mediante i prodotti numerici

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\large\color{White}\mathbf{M}_1=(a_{11}&plus;a_{22})(b_{11}&plus;b_{22})">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\large\mathbf{M}_1=(a_{11}&plus;a_{22})(b_{11}&plus;b_{22})">
  <img src="https://latex.codecogs.com/svg.image?\large\mathbf{M}_1=(a_{11}&plus;a_{22})(b_{11}&plus;b_{22})">
</picture>
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\large\color{White}\mathbf{M}_2=(a_{21}&plus;a_{22})b_{11}">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\large\mathbf{M}_2=(a_{21}&plus;a_{22})b_{11}">
  <img src="https://latex.codecogs.com/svg.image?\large\mathbf{M}_2=(a_{21}&plus;a_{22})b_{11}">
</picture>
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\large\color{White}\mathbf{M}_3=a_{11}(b_{12}-b_{22})">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\large\mathbf{M}_3=a_{11}(b_{12}-b_{22})">
  <img src="https://latex.codecogs.com/svg.image?\large\mathbf{M}_3=a_{11}(b_{12}-b_{22})">
</picture>
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\large\color{White}\mathbf{M}_4=a_{22}(b_{21}-b_{11})">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\large\mathbf{M}_4=a_{22}(b_{21}-b_{11})">
  <img src="https://latex.codecogs.com/svg.image?\large\mathbf{M}_4=a_{22}(b_{21}-b_{11})">
</picture>
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\large\color{White}\mathbf{M}_5=(a_{11}&plus;a_{12})b_{22}">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\large\mathbf{M}_5=(a_{11}&plus;a_{12})b_{22}">
  <img src="https://latex.codecogs.com/svg.image?\large\mathbf{M}_5=(a_{11}&plus;a_{12})b_{22}">
</picture>
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\large\color{White}\mathbf{M}_6=(a_{21}-a_{11})(b_{11}&plus;b_{12})">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\large\mathbf{M}_6=(a_{21}-a_{11})(b_{11}&plus;b_{12})">
  <img src="https://latex.codecogs.com/svg.image?\large\mathbf{M}_6=(a_{21}-a_{11})(b_{11}&plus;b_{12})">
</picture>
<br>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\large\color{White}\mathbf{M}_7=(a_{12}-a_{22})(b_{21}&plus;b_{22})">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\large\mathbf{M}_7=(a_{12}-a_{22})(b_{21}&plus;b_{22})">
  <img src="https://latex.codecogs.com/svg.image?\large\mathbf{M}_7=(a_{12}-a_{22})(b_{21}&plus;b_{22})">
</picture>


dove in questo caso banale le due matrici di input sono


<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\large\color{White}\mathbf{A}=\begin{pmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\end{pmatrix}\qquad\mathbf{B}=\begin{pmatrix}b_{11}&b_{12}\\b_{21}&b_{22}\end{pmatrix}">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\large\mathbf{A}=\begin{pmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\end{pmatrix}\qquad\mathbf{B}=\begin{pmatrix}b_{11}&b_{12}\\b_{21}&b_{22}\end{pmatrix}">
  <img src="https://latex.codecogs.com/svg.image?\large\mathbf{A}=\begin{pmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\end{pmatrix}\qquad\mathbf{B}=\begin{pmatrix}b_{11}&b_{12}\\b_{21}&b_{22}\end{pmatrix}">
</picture>
<br>
Siccome l'algoritmo è del tipo divide et impera, si ha un caso banale dove le due matrici <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\color{White}\mathbf{A}">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\mathbf{A}">
  <img src="https://latex.codecogs.com/svg.image?\mathbf{A}">
</picture> e <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\color{White}\mathbf{B}">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\mathbf{B}">
  <img src="https://latex.codecogs.com/svg.image?\mathbf{B}">
</picture> hanno dimensione <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\color{White}n=2">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?n=2">
  <img src="https://latex.codecogs.com/svg.image?n=2">
</picture>, mentre il caso ricorsivo è dato da <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\color{White}n>2">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?n>2">
  <img src="https://latex.codecogs.com/svg.image?n>2">
</picture>, per cui avviene la decomposizione.

## Differenze tra l'algoritmo di Strassen e l'algoritmo standard
|Caratteristiche|Naive|Strassen |
|--|--|--|
|Complessità di tempo|<picture><source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\color{White}O(n^3)"><source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?O(n^3)"><img src="https://latex.codecogs.com/svg.image?O(n^3)"></picture>|<picture><source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\color{White}O(n^{log_27})"><source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?O(n^{log_27})"><img src="https://latex.codecogs.com/svg.image?O(n^{log_27})"></picture>|
Approccio algoritmico|Iterativo (o divide et impera)|Divide et impera
Applicazione|Qualsiasi matrice|Matrici quadrate

Sebbene in linea teorica l'algoritmo di Strassen dimostri la non ottimalità dell'algoritmo standard per il prodotto tra due matrici, questo non trova pienamente riscontro in un ambiente computazionale. La creazione delle matrici temporanee <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?\color{White}\mathbf{M}_1%2C\ldots%2C\mathbf{M}_7">
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?\mathbf{M}_1%2C\ldots%2C\mathbf{M}_7">
  <img src="https://latex.codecogs.com/svg.image?\mathbf{M}_1%2C\ldots%2C\mathbf{M}_7">
</picture> richiede maggiore memoria, oltre al generare maggiore overhead per via dell'allocazione (e deallocazione) di tali matrici.

Nella [relazione](./relazione.pdf) viene approfondito lo sviluppo di un algoritmo capace di mitigare le problematiche intrinseche dell'algoritmo di Strassen.
