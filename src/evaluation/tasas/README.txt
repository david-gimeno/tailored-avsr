Para compilarlos:

gcc -o tasas tasas.c -Wall
gcc -o tasasIntervalo tasasIntervalo.c -Wall -lm

Te saldran avisos de compilacion pero no es problema.

Para usarlos, tendras en un fichero de texto las secuencias de palabras de referencia y salida separadas por #, cada una en una linea. Es decir:

ref1 ref2 .... refN#sal1 sal2 ... salM

Y ejecutaras:

tasas -s " " -f "#" -ie <nomfich>
tasasIntervalo -s " " -f "#" -ie <nomfich>

El primero te da el WER tal cual (debera coincidir, mas o menos, con lo que te dan las herramientas de Kaldi). 
El segundo te da el WER con los intervalos de confianza (el WER que da no es exactamente el de "tasas", porque es la media de los WER del bootstrapping)
