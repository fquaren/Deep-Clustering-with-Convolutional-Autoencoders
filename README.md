Allora, io quello che voglio farti fare è implementare una rete neurale convoluzionale (CNN) tipo U-Net per fare autoencoding.
Ossia, voglio che la rete prenda in ingresso un'immagine e generi la stessa immagine in uscita.
La U-Net è una rete pensata per la segmentazione di immagini, che nel ramo di sinistra estrae features dalle immagini mediante filtri successivi, mentre nel ramo di destra ottiene una segmentazione dell'immagine a partire dalle features estratte.
Io la voglio usare per fare autoencoding, con l'idea di utilizzare le features estratte per fare un clustering non supervisionato delle immagini nelle 3 classi TAC, RM e PET.
In ingresso alla U-Net non voglio passare l'intera immagine ma un patch, ossia una porzione di es 20x20 pixel.

Inizio a mandarti:
- articolo che spiega cosa è la U-Net
- articolo che usa le CNN per fare autoencoding su patch di immagini, ma in un ambito molto diverso (coregistrazione)
- articolo generale su utilizzo CNN nell'imaging medicale (per cultura)

Tu per iniziare a fare qualche prova prendi pure una porzione 20x20 nel centro di una delle immagini che ti ho mandato.
Appena ho un attimo di tempo ti mando poi set di immagini aggiornati, ossia pre-processati in modo che volume TAC, RM e PET siano all'interno dello stesso sistema di riferimento (ossia coregistrati).
