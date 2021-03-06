# Algoritmo di Strassen in ambiente GPU-CUDA
Il progetto sviluppato propone una strategia per l'accelerazione del prodotto tra due matrici in ambiente GPGPU. L'algoritmo presente ricorre all'algoritmo di Strassen per ridurre il tempo computazionale dell'algoritmo, studiandone anche le peculiarità.
La strategia di parallelizzazione è delegata alla libreria cuBLAS, che svolge le operazioni di addizione e sottrazione matriciale.

Durante la progettazione, è stata ideata in un primo momento un'implementazione in ambiente CPU, per poi "trasporre" la logica in ambiente GPGPU, cercando di mantenere quanto più uniforme possibile i due codici.

## Definizione algoritmo di Strassen
Siano date due matrici quadrate <img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{A}" title="\bg_white \mathbf{A}" /> e <img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{B}" title="\bg_white \mathbf{B}" /> di dimensioni <img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;n&space;\times&space;n" title="\bg_white n \times n" />. Si vuole calcolare la matrice <img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{C}&space;=&space;\mathbf{A}&space;\times&space;\mathbf{B}" title="\bg_white \mathbf{C} = \mathbf{A} \times \mathbf{B}" /> con l'algoritmo di Strassen.
L’algoritmo ricava la matrice risultato definendo sette nuove matrici

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{M}_1&space;=&space;(\mathbf{A}_{11}&space;&plus;&space;\mathbf{A}_{22})(\mathbf{B}_{11}&space;&plus;&space;\mathbf{B}_{22})" title="\bg_white \mathbf{M}_1 = (\mathbf{A}_{11} + \mathbf{A}_{22})(\mathbf{B}_{11} + \mathbf{B}_{22})" />

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{M}_2&space;=&space;(\mathbf{A}_{21}&space;&plus;&space;\mathbf{A}_{22})\mathbf{B}_{11}" title="\bg_white \mathbf{M}_2 = (\mathbf{A}_{21} + \mathbf{A}_{22})\mathbf{B}_{11}" />

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{M}_3&space;=&space;\mathbf{A}_{11}(\mathbf{B}_{12}&space;-&space;\mathbf{B}_{22})" title="\bg_white \mathbf{M}_3 = \mathbf{A}_{11}(\mathbf{B}_{12} - \mathbf{B}_{22})" />

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{M}_4&space;=&space;\mathbf{A}_{22}(\mathbf{B}_{21}&space;-&space;\mathbf{B}_{11})" title="\bg_white \mathbf{M}_4 = \mathbf{A}_{22}(\mathbf{B}_{21} - \mathbf{B}_{11})" />

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{M}_5&space;=&space;(\mathbf{A}_{11}&space;&plus;&space;\mathbf{A}_{12})\mathbf{B}_{22}" title="\bg_white \mathbf{M}_5 = (\mathbf{A}_{11} + \mathbf{A}_{12})\mathbf{B}_{22}" />

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{M}_6&space;=&space;(\mathbf{A}_{21}&space;-&space;\mathbf{A}_{11})(\mathbf{B}_{11}&space;&plus;&space;\mathbf{B}_{12})" title="\bg_white \mathbf{M}_6 = (\mathbf{A}_{21} - \mathbf{A}_{11})(\mathbf{B}_{11} + \mathbf{B}_{12})" />

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{M}_7&space;=&space;(\mathbf{A}_{12}&space;-&space;\mathbf{A}_{22})(\mathbf{B}_{21}&space;&plus;&space;\mathbf{B}_{22})" title="\bg_white \mathbf{M}_7 = (\mathbf{A}_{12} - \mathbf{A}_{22})(\mathbf{B}_{21} + \mathbf{B}_{22})" />

che riduce, di fatto, la complessità computazionale dell'algoritmo a <img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;O(n^{\log_2&space;7})" title="\bg_white O(n^{\log_2 7})" /> in quanto da 8 prodotti ne vengono computati "solo" 7. Successivamente si ricombina nella matrice risultato

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;&space;&space;\begin{pmatrix}&space;&space;&space;\mathbf{C}_{11}&space;&&space;\mathbf{C}_{12}\\&space;&space;&space;\mathbf{C}_{21}&space;&&space;\mathbf{C}_{22}&space;&space;\end{pmatrix}&space;&space;=&space;&space;\begin{pmatrix}&space;&space;&space;\mathbf{M}_1&space;&plus;&space;\mathbf{M}_4&space;-&space;\mathbf{M}_5&space;&plus;&space;\mathbf{M}_7&space;&&space;\mathbf{M}_3&space;&plus;&space;\mathbf{M}_5\\&space;&space;&space;\mathbf{M}_2&space;&plus;&space;\mathbf{M}_4&space;&&space;\mathbf{M}_1&space;-&space;\mathbf{M}_2&space;&plus;&space;\mathbf{M}_3&space;&plus;&space;\mathbf{M}_6&space;&space;\end{pmatrix}" title="\bg_white \begin{pmatrix} \mathbf{C}_{11} & \mathbf{C}_{12}\\ \mathbf{C}_{21} & \mathbf{C}_{22} \end{pmatrix} = \begin{pmatrix} \mathbf{M}_1 + \mathbf{M}_4 - \mathbf{M}_5 + \mathbf{M}_7 & \mathbf{M}_3 + \mathbf{M}_5\\ \mathbf{M}_2 + \mathbf{M}_4 & \mathbf{M}_1 - \mathbf{M}_2 + \mathbf{M}_3 + \mathbf{M}_6 \end{pmatrix}" />

Il "trucco" dell'algoritmo consiste nell'evitare il prodotto riga per colonna (algoritmo standard del prodotto matriciale) matrici, andando a ridurre il problema fintantoché le sette matrici non siano ricavabili mediante i prodotti numerici

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{M}_1&space;=&space;(a_{11}&space;&plus;&space;a_{22})(b_{11}&space;&plus;&space;b_{22})" title="\bg_white \mathbf{M}_1 = (a_{11} + a_{22})(b_{11} + b_{22})" />

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{M}_2&space;=&space;(a_{21}&space;&plus;&space;a_{22})b_{11}" title="\bg_white \mathbf{M}_2 = (a_{21} + a_{22})b_{11}" />

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{M}_3&space;=&space;a_{11}(b_{12}&space;-&space;b_{22})" title="\bg_white \mathbf{M}_3 = a_{11}(b_{12} - b_{22})" />

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{M}_4&space;=&space;a_{22}(b_{21}&space;-&space;b_{11})" title="\bg_white \mathbf{M}_4 = a_{22}(b_{21} - b_{11})" />

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{M}_5&space;=&space;(a_{11}&space;&plus;&space;a_{12})b_{22}" title="\bg_white \mathbf{M}_5 = (a_{11} + a_{12})b_{22}" />

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{M}_6&space;=&space;(a_{21}&space;-&space;a_{11})(b_{11}&space;&plus;&space;b_{12})" title="\bg_white \mathbf{M}_6 = (a_{21} - a_{11})(b_{11} + b_{12})" />

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{M}_7&space;=&space;(a_{12}&space;-&space;a_{22})(b_{21}&space;&plus;&space;b_{22})" title="\bg_white \mathbf{M}_7 = (a_{12} - a_{22})(b_{21} + b_{22})" />

dove in questo caso banale le due matrici di input sono

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;&space;&space;\mathbf{A}&space;=&space;\begin{pmatrix}&space;&space;&space;a_{11}&space;&&space;a_{12}\\&space;&space;&space;a_{21}&space;&&space;a_{22}&space;&space;\end{pmatrix}&space;&space;\qquad&space;&space;&space;&space;\mathbf{B}&space;=&space;\begin{pmatrix}&space;&space;&space;b_{11}&space;&&space;b_{12}\\&space;&space;&space;b_{21}&space;&&space;b_{22}&space;&space;\end{pmatrix}" title="\bg_white \mathbf{A} = \begin{pmatrix} a_{11} & a_{12}\\ a_{21} & a_{22} \end{pmatrix} \qquad \mathbf{B} = \begin{pmatrix} b_{11} & b_{12}\\ b_{21} & b_{22} \end{pmatrix}" />

Siccome l'algoritmo è del tipo divide et impera, si ha un caso banale dove le due matrici <img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{A}" title="\bg_white \mathbf{A}" /> e <img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{B}" title="\bg_white \mathbf{B}" /> hanno dimensione <img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;n&space;=&space;2" title="\bg_white n = 2" />, mentre il caso ricorsivo è dato da <img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;n&space;=&space;2" title="\bg_white n > 2" />, per cui avviene la decomposizione.

## Differenze tra l'algoritmo di Strassen e l'algoritmo standard
|Caratteristiche|Naive|Strassen |
|--|--|--|
|Complessità di tempo|<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;O(n^3)" title="\bg_white O(n^3)" />|<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;O(n^{\log_2&space;7})" title="\bg_white O(n^{\log_2 7})" />|
Approccio algoritmico|Iterativo (o divide et impera)|Divide et impera
Applicazione|Qualsiasi matrice|Matrici quadrate

Sebbene in linea teorica l'algoritmo di Strassen dimostri la non ottimalità dell'algoritmo standard per il prodotto tra due matrici, questo non trova pienamente riscontro in un ambiente computazionale. La creazione delle matrici temporanee <img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathbf{M}_1,&space;\ldots,&space;\mathbf{M}_7" title="\bg_white \mathbf{M}_1, \ldots, \mathbf{M}_7" /> richiede maggiore memoria, oltre al generare maggiore overhead per via dell'allocazione (e deallocazione) di tali matrici.
