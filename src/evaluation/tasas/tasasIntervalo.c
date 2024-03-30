#define reemplazamiento 1
#define insercion 2
#define borrado 3
#define fin -1
#define OPP "-p"
#define OPAYUDA "-h"
#define OPVERBOSO "-v"
#define OPFICHSIM "-w"
#define OPSEPFRASES "-f"
#define OPSEPSIMBOLOS "-s"
#define OPCONFCOMP "-C"
#define OPCONFNONULA "-c"
#define OPMAXITER "-n"				
#define PRE "-pre"
#define PRA "-pra"
#define PA "-pa"
#define IP "-ip"
#define IE "-ie"
#define PSB "-psb"
#define IEP "-iep"
#define IAP "-iap"

/*26:*/
//#line 989 "tasas.w"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <math.h>
/*2:*/
//#line 64 "tasas.w"

//#line 65 "tasas.w"
typedef struct{
  int talla;
  int*simbolo;
}tipo_cadena;

/*:2*/
//#line 994 "tasas.w"

/*3:*/
//#line 71 "tasas.w"

typedef struct{
  int talla;
  tipo_cadena*cadena;
}lista_de_cadenas;

/*:3*/
//#line 995 "tasas.w"

/*24:*/
//#line 711 "tasas.w"

typedef struct forward_nodo_diccionario{
  char*simbolo;
  struct forward_nodo_diccionario*siguiente;
}nodo_diccionario;

nodo_diccionario*diccionario= NULL;
int talla_diccionario= 0;
int diccionario_bloqueado= 0;

int identificador_simbolo(char*simbolo){
  int i;
  nodo_diccionario*iterador,*previo;
  
  if(diccionario==NULL){
    if(diccionario_bloqueado){
      fprintf(stderr,
	      "Error: el simbolo %s no aparece en la declaracion inicial.\n",
	      simbolo);
      exit(-1);
    }
    diccionario= malloc(sizeof(nodo_diccionario));
    diccionario->siguiente= NULL;
    diccionario->simbolo= malloc((strlen(simbolo)+1)*sizeof(char));
    strcpy(diccionario->simbolo,simbolo);
    talla_diccionario= 1;
    return talla_diccionario;
  }
  else{
    previo= NULL;
    for(iterador= diccionario,i= 0;iterador;iterador= iterador->siguiente,i++){
      if(strcmp(iterador->simbolo,simbolo)==0)return i+1;
      previo= iterador;
    }
    if(diccionario_bloqueado){
      fprintf(stderr,
	      "Error: el simbolo %s no aparece en la declaracion inicial.\n",
	      simbolo);
      exit(-1);
    }
    iterador= previo->siguiente= malloc(sizeof(nodo_diccionario));
    iterador->simbolo= malloc((strlen(simbolo)+1)*sizeof(char));
    strcpy(iterador->simbolo,simbolo);
    iterador->siguiente= NULL;
    return++talla_diccionario;
  }
}

char*simbolo_identificador(int id){
  nodo_diccionario*iterador;
  int i= 1;
  
  for(iterador= diccionario;iterador;iterador= iterador->siguiente)
    if(i++==id)return iterador->simbolo;
  return"!!!";
}

void libera_memoria_diccionario(){
  nodo_diccionario*iterador,*aux;
  
  for(iterador= diccionario;iterador;iterador= aux){
    aux= iterador->siguiente;
    free(iterador->simbolo);
    free(iterador);
  }
}

void bloquea_diccionario(){
  diccionario_bloqueado= 1;
}

lee_declaracion_de_simbolos(char*fichero){
  FILE*fp;
  char simbolo[1024];
  
  if(strcmp(fichero,"-")==0)fp= stdin;
  else fp= fopen(fichero,"r");
  if(fp==NULL){
    fprintf(stderr,"Fichero %s inexistente\n",fichero);
    exit(-1);
  }
  while(fscanf(fp,"%s",simbolo)>0)identificador_simbolo(simbolo);
  if(strcmp(fichero,"-")!=0)fclose(fp);
  bloquea_diccionario();
}


/*:24*/
//#line 996 "tasas.w"

/*25:*/
//#line 803 "tasas.w"

typedef struct tcelda{
  int simbolo;
  int valor;
  struct tcelda*sig;
}celda;

struct{
  int dim;
  celda**c;
}mconf;

struct{
  celda*ptr;
  int fila,col;
}mconf_it;

mconf_malloc(int dim){
  int i;
  
  mconf.dim= dim;
  mconf.c= malloc((dim+1)*sizeof(celda*));
  for(i= 0;i<=dim;i++)
    mconf.c[i]= NULL;
}

mconf_free(){
  int i;
  celda*p,*q;
  
  for(i= 0;i<=mconf.dim;i++){
    for(p= mconf.c[i],p&&(q= p->sig);p;p= q,q&&(q= q->sig)){
      free(p);
    }
}
  free(mconf.c);
  mconf.dim= 0;
}

mconf_reinicia(){
  int i;
  celda*p;
  
  for(i= 0;i<=mconf.dim;i++)
    for(p= mconf.c[i];p;p= p->sig)
      p->valor= 0;
}


int mconf_ref(int i,int j){
  if(mconf_it.fila!=i||mconf_it.col>j){
    mconf_it.fila= i;
    mconf_it.col= 0;
    mconf_it.ptr= mconf.c[i];
  }
  while(mconf_it.ptr&&(mconf_it.ptr->simbolo<j)){
    mconf_it.ptr= mconf_it.ptr->sig;
  }
  if(mconf_it.ptr&&(mconf_it.ptr->simbolo==j))
    return mconf_it.ptr->valor;
  return 0;
}

void mconf_incr(int i,int j){
  celda*p,*q;
  
  mconf_it.fila= -1;
  if(mconf.c[i]){
    for(p= mconf.c[i],q= NULL;p;q= p,p= p->sig){
      if(p->simbolo==j){
	p->valor++;
	break;
      }
      else if(p->simbolo>j){
	if(q){
	  q->sig= malloc(sizeof(celda));
	  q->sig->simbolo= j;
	  q->sig->valor= 1;
	  q->sig->sig= p;
	  break;
	}
	else{
	  mconf.c[i]= malloc(sizeof(celda));
	  mconf.c[i]->simbolo= j;
	  mconf.c[i]->valor= 1;
	  mconf.c[i]->sig= p;
	  break;
	}
      }
    }
    if(p==NULL){
      q->sig= malloc(sizeof(celda));
      q->sig->simbolo= j;
      q->sig->valor= 1;
      q->sig->sig= NULL;
    }
  }
  else{
    mconf.c[i]= malloc(sizeof(celda));
    mconf.c[i]->simbolo= j;
    mconf.c[i]->valor= 1;
    mconf.c[i]->sig= NULL;
  }
}

mconf_imprime_completa(char*fichero){
  nodo_diccionario*iterador;
  int i,j,k,long_pal_mas_larga,max_num;
  char aux[64],fmt1[64],fmt2[64],fmt3[64];
  int ancho1,ancho3;
  FILE*fp;
  
  long_pal_mas_larga= 0;
  for(iterador= diccionario;iterador!=NULL;iterador= iterador->siguiente)
    if(long_pal_mas_larga<strlen(iterador->simbolo))
      long_pal_mas_larga= strlen(iterador->simbolo);
  sprintf(aux,"%d",talla_diccionario);
  ancho1= strlen(aux);
  sprintf(fmt1,"%%%dd",ancho1);
  sprintf(fmt2," %%-%ds",long_pal_mas_larga);
  
  max_num= 0;
  for(i= 0;i<=mconf.dim;i++)
    for(j= 0;j<=mconf.dim;j++)
      if(max_num<mconf_ref(i,j))
	max_num= mconf_ref(i,j);
  
  sprintf(aux,"%d",max_num);
  ancho3= strlen(aux);
  sprintf(fmt3," %%%dd",ancho3);
  
  if(strcmp(fichero,"-")==0)fp= stdout;
  else fp= fopen(fichero,"w");
  
  for(k= 0;k<ancho1+1+long_pal_mas_larga;k++)fprintf(fp," ");
  for(j= 0;j<=talla_diccionario;j++)
    fprintf(fp,fmt3,j);
  fprintf(fp,"\n");
  
  fprintf(fp,fmt1,0);
  fprintf(fp,fmt2,"");
  
  for(j= 0;j<=talla_diccionario;j++)
    fprintf(fp,fmt3,mconf_ref(0,j));
  fprintf(fp,"\n");
  
  for(i= 1,iterador= diccionario;iterador!=NULL;i++,iterador= iterador->siguiente){
    fprintf(fp,fmt1,i);
    fprintf(fp,fmt2,iterador->simbolo);
    for(j= 0;j<=talla_diccionario;j++)
      fprintf(fp,fmt3,mconf_ref(i,j));
    fprintf(fp,"\n");
  }
  if(strcmp(fichero,"-")!=0)
    fclose(fp);
}

mconf_imprime_elementos_no_nulos(char*fichero){
  int i;
  FILE*fp;
  celda*p;
  
  if(strcmp(fichero,"-")==0)fp= stdout;
  else fp= fopen(fichero,"w");
  for(i= 0;i<=talla_diccionario;i++){
    for(p= mconf.c[i];p;p= p->sig){
      if(p->valor!=0){
	if(i==0)fprintf(fp,"*  -> %s : %d\n",
			simbolo_identificador(p->simbolo),p->valor);
	else if(p->simbolo==0)fprintf(fp,"*  %s -> : %d\n",
				      simbolo_identificador(i),p->valor);
	else{
	  if(i!=p->simbolo)fprintf(fp,"* ");
	  else fprintf(fp,"  ");
	  fprintf(fp," %s -> %s : %d\n",
		  simbolo_identificador(i),simbolo_identificador(p->simbolo),
		  p->valor);
	}
      }
    }
  }
  if(strcmp(fichero,"-")!=0)fclose(fp);
}

/*:25*/
//#line 997 "tasas.w"

/*6:*/
//#line 257 "tasas.w"

int**va;
double**d,**dd;

/*:6*/
//#line 998 "tasas.w"

/*5:*/
//#line 199 "tasas.w"

void gp(double p,tipo_cadena c,tipo_cadena s,
	int*ns,int*ni,int*nb,int*na,
	int conf){
  double dsa,di,db;
  double gs,gi,gb,ga;
  int i,j;
  
  gs= 1.0;gi= gb= p;ga= 0.0;
  *ns= *ni= *nb= *na= 0;
  va[0][0]= fin;d[0][0]= 0.0;
  for(i= 1;i<=c.talla;i++){
    d[i][0]= d[i-1][0]+gb;
    va[i][0]= borrado;
  }
  for(j= 1;j<=s.talla;j++){
    d[0][j]= d[0][j-1]+gi;
    va[0][j]= insercion;
  }
  for(i= 1;i<=c.talla;i++){
    for(j= 1;j<=s.talla;j++){
      dsa = d[i-1][j-1]+((c.simbolo[i]==s.simbolo[j])?ga:gs);
      di = d[i][j-1]+gi;db= d[i-1][j]+gb;
      if(dsa<=di){
	if(dsa<=db){
	  d[i][j]= dsa;
	  va[i][j]= reemplazamiento;
	}
	else{
	  d[i][j]= db;
	  va[i][j]= borrado;
	}
      }
      else{
	if(di<db){
	  d[i][j]= di;
	  va[i][j]= insercion;
	}
	else{
	  d[i][j]= db;
	  va[i][j]= borrado;
	}
      }
    }
  }
  i= c.talla;j= s.talla;
  do{
    switch(va[i][j]){
    case reemplazamiento:
      if(conf)mconf_incr(c.simbolo[i],s.simbolo[j]);
      if(c.simbolo[i]==s.simbolo[j])(*na)++;
      else(*ns)++;
      i--;j--;
      break;
    case insercion:
      if(conf)mconf_incr(0,s.simbolo[j]);
      (*ni)++;
      j--;
      break;
    case borrado:
if(conf)mconf_incr(c.simbolo[i],0);
 (*nb)++;
 i--;
 break;
    }
  }while(va[i][j]!=fin);
}

/*:5*/
//#line 999 "tasas.w"

/*9:*/
//#line 284 "tasas.w"

void Gp(double p,lista_de_cadenas C,lista_de_cadenas S,
	int*ns,int*ni,int*nb,int*na,int conf){
  int i,nns,nni,nnb,nna;
  
  *ns= *ni= *nb= *na= 0;
  for(i= 0;i<C.talla;i++){
    gp(p,C.cadena[i],S.cadena[i],&nns,&nni,&nnb,&nna,conf);
    *ns+= nns;
    *ni+= nni;
    *nb+= nnb;
    *na+= nna;
  }
}

/*:9*/
//#line 1000 "tasas.w"

/*10:*/
//#line 323 "tasas.w"

double tasa_generica(int ns,int ni,int nb,int na,
		     double alphas,double alphai,double alphab,double alphaa,
		     double betas,double betai,double betab,double betaa){
  
  return 100.0*(alphas*ns+alphai*ni+alphab*nb+alphaa*na)/
    (betas*ns+betai*ni+betab*nb+betaa*na);
}

/*:10*/
//#line 1001 "tasas.w"

/*12:*/
//#line 337 "tasas.w"

double tasa_pra(int ns,int ni,int nb,int na){
  return tasa_generica(ns,ni,nb,na,
		       0.0,0.0,0.0,1.0,
		       1.0,1.0,1.0,1.0);
}

/*:12*//*13:*/
//#line 345 "tasas.w"

double tasa_pre(int ns,int ni,int nb,int na){
  return tasa_generica(ns,ni,nb,na,
		       1.0,1.0,1.0,0.0,
		       1.0,1.0,1.0,1.0);
}

/*:13*//*14:*/
//#line 353 "tasas.w"

double tasa_pa(int ns,int ni,int nb,int na){
  return tasa_generica(ns,ni,nb,na,
		       0.0,0.0,0.0,1.0,
		       1.0,0.0,1.0,1.0);
}

/*:14*//*15:*/
//#line 361 "tasas.w"

double tasa_ip(int ns,int ni,int nb,int na){
  return tasa_generica(ns,ni,nb,na,
		       0.0,-1.0,0.0,1.0,
		       1.0,0.0,1.0,1.0);
}

/*:15*//*16:*/
//#line 369 "tasas.w"

double tasa_ie(int ns,int ni,int nb,int na){
  return tasa_generica(ns,ni,nb,na,
		       1.0,1.0,1.0,0.0,
		       1.0,0.0,1.0,1.0);
}

/*:16*//*17:*/
//#line 377 "tasas.w"

double tasa_psb(int ns,int ni,int nb,int na){
  return tasa_generica(ns,ni,nb,na,
		       1.0,0.0,1.0,0.0,
		       1.0,0.0,1.0,1.0);
}
/*:17*//*18:*/
//#line 384 "tasas.w"

double tasa_iep(int ns,int ni,int nb,int na){
  return tasa_generica(ns,ni,nb,na,
		       1.0,0.5,0.5,0.0,
		       1.0,0.0,1.0,1.0);
}

/*:18*//*19:*/
//#line 392 "tasas.w"

double tasa_iap(int ns,int ni,int nb,int na){
  return tasa_generica(ns,ni,nb,na,
		       0.0,-0.5,0.5,1.0,
		       1.0,0.0,1.0,1.0);
}

/*:19*/
//#line 1002 "tasas.w"

/*20:*/
//#line 454 "tasas.w"

/*21:*/
//#line 492 "tasas.w"

double inicializa_lambda(lista_de_cadenas C,lista_de_cadenas S){
  int ns,ni,nb,na;
  double dsa,di,db;
  double ddsa,ddi,ddb;
  double fsa,fi,fb;
  double gs,gi,gb,ga;
  int i,j,n;
  tipo_cadena c,s;
  
  gs= 1.0;gi= gb= 1.0;ga= 0.0;
  ns= ni= nb= na= 0;
  for(n= 0;n<C.talla;n++){
    c= C.cadena[n];
    s= S.cadena[n];
    va[0][0]= fin;
    d[0][0]= 0.0;
    dd[0][0]= 0.0;
    for(i= 1;i<=c.talla;i++){
      d[i][0]= d[i-1][0]+gb;
      dd[i][0]= dd[i-1][0]+1;
      va[i][0]= borrado;
    }
    for(j= 1;j<=s.talla;j++){
      d[0][j]= d[0][j-1]+gi;
      dd[0][j]= dd[0][j-1]+1;
      va[0][j]= insercion;
    }
    for(i= 1;i<=c.talla;i++){
      for(j= 1;j<=s.talla;j++){
	dsa= d[i-1][j-1]+((c.simbolo[i]==s.simbolo[j])?ga:gs);
	di= d[i][j-1]+gi;
	db= d[i-1][j]+gb;
	ddsa= dd[i-1][j-1]+1;
	ddi= dd[i][j-1]+1;ddb= dd[i-1][j]+1;
	fsa= dsa/ddsa;
	fi= di/ddi;
	fb= db/ddb;
	if(fsa<=fi){
	  if(fsa<=fb){
	    d[i][j]= dsa;
	    dd[i][j]= ddsa;
	    va[i][j]= reemplazamiento;
	  }
	  else{
	    d[i][j]= db;
	    dd[i][j]= ddb;
	    va[i][j]= borrado;
	  }
	}else{
	  if(fi<fb){
	    d[i][j]= di;
	    dd[i][j]= ddi;
	    va[i][j]= insercion;
	  }
	  else{
	    d[i][j]= db;
	    dd[i][j]= ddb;
	    va[i][j]= borrado;
	  }
	}
      }
    }
    
    i= c.talla;j= s.talla;
    do{
      switch(va[i][j]){
      case reemplazamiento:
	if(c.simbolo[i]==s.simbolo[j])na++;
	else ns++;
	i--;j--;
	break;
      case insercion:
	ni++;
	j--;
	break;
      case borrado:
	nb++;
	i--;
	break;
      }
    }while(va[i][j]!=fin);
  }
  return(ns+ni+nb)/(double)(ns+ni+nb+na);
}

/*:21*/
//#line 455 "tasas.w"

double Fp(lista_de_cadenas C,lista_de_cadenas S,
	  int*ns,int*ni,int*nb,int*na,int conf){
  double lambda,lambdacero;
  double p;
  
  lambda= inicializa_lambda(C,S);
  do{
    lambdacero= lambda;
    p= 1.0-lambdacero/2.0;
    if(conf)mconf_reinicia();
    Gp(p,C,S,ns,ni,nb,na,conf);
    lambda= (*ns+*ni+*nb)/(double)(*ni+*nb+*ns+*na);
  }while(abs(lambda-lambdacero)>0.000001);
  return p;
}

/*:20*/
//#line 1003 "tasas.w"

/*30:*/
//#line 1235 "tasas.w"

void ayuda_y_aborta(char*argv[]){
  fprintf(stderr,"Evaluador de sistema RAH\n");
  fprintf(stderr,"Uso: %s fich [%s \"c\"] [%s #|%s \"c\"]",
	  argv[0],OPSEPFRASES,OPSEPSIMBOLOS,OPSEPSIMBOLOS);
  fprintf(stderr," [%s #] [TASA] [%s mat|%s mat] [%s dicc] [%s]\n",
	  OPP,OPCONFNONULA,OPCONFCOMP,OPFICHSIM,OPVERBOSO);
  fprintf(stderr,"donde: \n");
  fprintf(stderr,"fich es el fichero de datos\n");
  fprintf(stderr,"  %s \"c\"   : hace que c separe frases en una linea\n",OPSEPFRASES);
  fprintf(stderr,"  %s #     : hace que cada # caracteres se consideren un simbolo\n",
	  OPSEPSIMBOLOS);
  fprintf(stderr,"  %s \"cad\" : caracts. en \"cad\" separan simbolos\n",OPSEPSIMBOLOS);
  fprintf(stderr,"  %s #     : fija el parametro p a #\n",OPP);
  fprintf(stderr,"  TASA     : proporciona una tasa concreta\n");
  fprintf(stderr,"             [%s|%s|%s|%s|%s|%s|%s|%s] \n",
	  PRA,PRE,PA,IP,IE,PSB,IEP,IAP);
  fprintf(stderr,"  %s mat   : guarda en fichero mat la matriz de confusion\n",OPCONFCOMP);
  fprintf(stderr,"  %s mat   : guarda en mat elementos no nulos de matriz de confusion\n",
	  OPCONFNONULA);
  fprintf(stderr,"  %s dicc  : toma del fichero dicc el orden de los simbolos \n",
	  OPFICHSIM);
  fprintf(stderr,"  %s num_iter  : numero de iteraciones de Bootstrap \n",OPMAXITER);
  fprintf(stderr,"  %s       : muestra el numero de ops. de cada tipo y valor de p usado\n",
	  OPVERBOSO);
  fprintf(stderr,"Si el nombre de un fichero es \"-\" se toma la entrada/salida estandar\n");
  fprintf(stderr,"POR DEFECTO: %s %s \"*\" %s 1 %s %s 1000\n",
	  argv[0],OPSEPFRASES,OPSEPSIMBOLOS,PRA,OPMAXITER);
  exit(-1);
}

/*:30*/
//#line 1004 "tasas.w"

/*22:*/
//#line 589 "tasas.w"

/*23:*/
//#line 644 "tasas.w"

void cadena_sin_separadores(char*linea,int desde,int hasta,int talla_simbolo,
			    tipo_cadena*c)
{
  int j,k;
  char simbolo[1024];
  
  c->talla= (hasta-desde)/talla_simbolo;
  c->simbolo= malloc((1+c->talla)*sizeof(int));
  for(j= 0;j<c->talla;j++){
    for(k= 0;k<talla_simbolo;k++)
      simbolo[k]= linea[j*talla_simbolo+k+desde];
    simbolo[k]= '\0';
    c->simbolo[j+1]= identificador_simbolo(simbolo);
  }
}

int es_separador(char caracter,char*separadores){
  int i,j;
  j= strlen(separadores);
  for(i= 0;i<j;i++)if(caracter==separadores[i])return 1;
  return 0;
}

void cadena_con_separadores(char*linea,int desde,int hasta,char*separadores,
			    tipo_cadena*c)
{
  int i,j,k;
  char simbolo[1024];

  c->talla= 0;
  for(j= desde;es_separador(linea[j],separadores);j++);
  while(j<hasta){
    while((j<hasta)&&!es_separador(linea[j],separadores))j++;
    while((j<hasta)&&es_separador(linea[j],separadores))j++;
    c->talla++;
  }
  if(c->talla>0){
    c->simbolo= malloc((1+c->talla)*sizeof(int));
    i= 1;
    for(j= desde;es_separador(linea[j],separadores);j++);
    while(j<hasta){
      k= 0;
      while((j<hasta)&&!es_separador(linea[j],separadores))
	simbolo[k++]= linea[j++];
      simbolo[k]= '\0';
      c->simbolo[i++]= identificador_simbolo(simbolo);
      while((j<hasta)&&es_separador(linea[j],separadores))j++;
    }
  }
}


/*:23*/
//#line 590 "tasas.w"

void lee_datos(char*fichero,lista_de_cadenas*C,lista_de_cadenas*S,
	       char separador_de_cadenas,int talla_simbolo,char*separador_de_simbolos){
  FILE*fp;
  char linea[2048];
  int j,talla_linea,fin_c,inicio_s,reserva_inicial;
  
  reserva_inicial= 512;
  C->cadena= malloc(reserva_inicial*sizeof(tipo_cadena));
  S->cadena= malloc(reserva_inicial*sizeof(tipo_cadena));
  
  if(strcmp(fichero,"-")==0)fp= stdin;
  else fp= fopen(fichero,"r");
  if(fp==NULL){
    fprintf(stderr,"Fichero %s inexistente.\n",fichero);
    exit(-1);
  }
  
  C->talla= S->talla= 0;
  while(fgets(linea,2048,fp)!=NULL){
    if(linea[strlen(linea)-1]=='\n')linea[strlen(linea)-1]= '\0';
    talla_linea= strlen(linea);
    for(j= 0;j<talla_linea;j++)if(linea[j]==separador_de_cadenas)break;
    if(j==talla_linea){
      fprintf(stderr,"No hay separador de cadenas (%c) en la linea %d.\n",
	      separador_de_cadenas,C->talla+1);
      fprintf(stderr,"\"%s\"\n",linea);
      exit(-1);
    }
    fin_c= j;inicio_s= j+1;
    if(talla_simbolo==0){
      cadena_con_separadores(linea,0,fin_c,separador_de_simbolos,
			     &C->cadena[C->talla]);
      cadena_con_separadores(linea,inicio_s,talla_linea,separador_de_simbolos,
			     &S->cadena[S->talla]);
      
    }
    else{
      cadena_sin_separadores(linea,0,fin_c,talla_simbolo,
			     &C->cadena[C->talla]);
      cadena_sin_separadores(linea,inicio_s,talla_linea,talla_simbolo,
			     &S->cadena[S->talla]);
    }
    C->talla++;S->talla++;
    if(C->talla>=reserva_inicial){
      C->cadena= realloc(C->cadena,reserva_inicial*sizeof(tipo_cadena)*2);
      S->cadena= realloc(S->cadena,reserva_inicial*sizeof(tipo_cadena)*2);
      reserva_inicial*= 2;
    }
  }
  if(strcmp(fichero,"-")!=0)fclose(fp);
}


/*:22*/
//#line 1005 "tasas.w"


int main(int argc,char*argv[]){
  lista_de_cadenas C,S, Cact,Sact;
  int i,max_s,max_c,talla_simbolo,ns,ni,nb,na;
  char fichero[256],fichero_simbolos[256],fichero_matriz[256],
    con_matriz,*tasa,normalizado,
    separador_de_cadenas,separador_de_simbolos[256];
  double p,rtasa;
  int verboso;
  
  int max_iter_boot=1000;
  /*29:*/
//#line 1165 "tasas.w"
  
  
  separador_de_cadenas= '*';
  talla_simbolo= 1;
  strcpy(fichero,"");
  strcpy(fichero_simbolos,"");
  strcpy(fichero_matriz,"");
  con_matriz= 0;
  tasa= PRE;
  verboso= 0;
  p= -1e30;
  if(argc==1)ayuda_y_aborta(argv);
  else{
    for(i= 1;i<argc;i++){
      if(strcmp(argv[i],OPAYUDA)==0){
	ayuda_y_aborta(argv);
      }
      if(strcmp(argv[i],OPP)==0){
	normalizado= 0;p= atof(argv[i+1]);i++;continue;
      }
      if(strcmp(argv[i],OPMAXITER)==0){
	max_iter_boot= atof(argv[i+1]);i++;continue;
      }
      if(strcmp(argv[i],OPFICHSIM)==0){
	strcpy(fichero_simbolos,argv[++i]);continue;
      }
      if(strcmp(argv[i],OPSEPFRASES)==0){
	separador_de_cadenas= argv[++i][0];continue;
      }
      if(strcmp(argv[i],OPVERBOSO)==0){
	verboso= 1;continue;
      }
      if(strcmp(argv[i],OPSEPSIMBOLOS)==0){
	if(atoi(argv[i+1])<=0){
	  talla_simbolo= 0;
	  strcpy(separador_de_simbolos,argv[++i]);
	}
	else{
	  talla_simbolo= atoi(argv[++i]);
	}
	continue;
      }
      if(strcmp(argv[i],PRA)==0||strcmp(argv[i],PRE)==0||
	 strcmp(argv[i],PA)==0||
	 strcmp(argv[i],IP)==0||strcmp(argv[i],IE)==0||
	 strcmp(argv[i],PSB)==0||
	 strcmp(argv[i],IEP)==0||strcmp(argv[i],IAP)==0){
	tasa= argv[i];
	continue;
      }
      if(strcmp(argv[i],OPCONFNONULA)==0){
	con_matriz= 1;
	strcpy(fichero_matriz,argv[++i]);
	continue;
      }
      if(strcmp(argv[i],OPCONFCOMP)==0){
	con_matriz= 2;
	strcpy(fichero_matriz,argv[++i]);
	continue;
      }
      if(strcmp(fichero,"")==0)
	strcpy(fichero,argv[i]);
      else 
	ayuda_y_aborta(argv);
    }
    if(strcmp(fichero,"")==0)
      ayuda_y_aborta(argv);
    if(p==-1e30){
      if(strcmp(tasa,PRA)==0||strcmp(tasa,PRE)==0)normalizado= 1;
      else{
	if(strcmp(tasa,PA)==0||strcmp(tasa,PSB)==0||strcmp(tasa,IEP)==0
	   ||strcmp(tasa,IAP)==0)p= 0.5;
	else p= 1;
	normalizado= 0;
      }
    }
  }
  
  
  /*:29*/
//#line 1016 "tasas.w"
  
  if(strcmp(fichero_simbolos,"")!=0)
    lee_declaracion_de_simbolos(fichero_simbolos);

  lee_datos(fichero,&C,&S,separador_de_cadenas,talla_simbolo,separador_de_simbolos);

  /*27:*/
//#line 1049 "tasas.w"
  
  max_s= 0;max_c= 0;
  for(i= 0;i<S.talla;i++)
    max_s= (max_s>S.cadena[i].talla)?max_s:S.cadena[i].talla;

  for(i= 0;i<C.talla;i++)
    max_c= (max_c>C.cadena[i].talla)?max_c:C.cadena[i].talla;
  
  /*:27*/
//#line 1021 "tasas.w"
  
  /*7:*/
//#line 262 "tasas.w"
  
  d= malloc((max_c+1)*sizeof(double*));
  d[0]= malloc((max_c+1)*(max_s+1)*sizeof(double));
  for(i= 1;i<=max_c;i++)
    d[i]= d[i-1]+max_s+1;

  dd= malloc((max_c+1)*sizeof(double*));
  dd[0]= malloc((max_c+1)*(max_s+1)*sizeof(double));
  for(i= 1;i<=max_c;i++)
    dd[i]= dd[i-1]+max_s+1;

  va= malloc((max_c+1)*sizeof(int*));
  va[0]= malloc((max_c+1)*(max_s+1)*sizeof(int));
  for(i= 1;i<=max_c;i++)
    va[i]= va[i-1]+max_s+1;
  
  /*:7*/
  //#line 1023 "tasas.w"

  /*InicializaciÃ³n de las cadenas de bootstrapping*/
  Cact.talla=C.talla;
  Cact.cadena= malloc(Cact.talla*sizeof(tipo_cadena));
  
  Sact.talla=S.talla;
  Sact.cadena= malloc(Sact.talla*sizeof(tipo_cadena));

  int aleatorio, x, iterBootStrap;
  float rtasaMedia=0, rtasaDesv=0;
  srand(time(0));

  for(iterBootStrap=0;iterBootStrap<max_iter_boot;iterBootStrap++){    
    for(x=0;x<Cact.talla;x++){
      aleatorio=rand() % Cact.talla;
      //      printf("alea %d\n",aleatorio);
	
      Cact.cadena[x]=C.cadena[aleatorio];
      Sact.cadena[x]=S.cadena[aleatorio];
    }
    
  

  if(con_matriz)mconf_malloc(talla_diccionario);
  if(normalizado){
    p= Fp(Cact,Sact,&ns,&ni,&nb,&na,con_matriz);
    if(strcmp(tasa,PRA)==0)rtasa= tasa_pra(ns,ni,nb,na);
    else if(strcmp(tasa,PRE)==0)rtasa= tasa_pre(ns,ni,nb,na);
  }
  else{
    Gp(p,Cact,Sact,&ns,&ni,&nb,&na,con_matriz);
    if(strcmp(tasa,PRE)==0)rtasa= tasa_pre(ns,ni,nb,na);
    else if(strcmp(tasa,PRA)==0)rtasa= tasa_pra(ns,ni,nb,na);
    else if(strcmp(tasa,PA)==0)rtasa= tasa_pa(ns,ni,nb,na);
    else if(strcmp(tasa,IP)==0)rtasa= tasa_ip(ns,ni,nb,na);
    else if(strcmp(tasa,IE)==0)rtasa= tasa_ie(ns,ni,nb,na);
    else if(strcmp(tasa,PSB)==0)rtasa= tasa_psb(ns,ni,nb,na);
    else if(strcmp(tasa,IEP)==0)rtasa= tasa_iep(ns,ni,nb,na);
    else if(strcmp(tasa,IAP)==0)rtasa= tasa_iap(ns,ni,nb,na);
  }
  /*28:*/
  //#line 1056 "tasas.w"
  
  rtasaMedia+=rtasa;
  rtasaDesv+=(rtasa*rtasa);
  //  printf("ue %lf\n",rtasa);
  
  }/*End del for iterBootStrap*/
  
  rtasaMedia=rtasaMedia/max_iter_boot;
  rtasaDesv=sqrt((rtasaDesv/max_iter_boot)-(rtasaMedia*rtasaMedia));

  printf("%lf +- %lf \n",rtasaMedia, 1.64*rtasaDesv);
  if(con_matriz==1)
    mconf_imprime_elementos_no_nulos(fichero_matriz);
  if(con_matriz==2)
    mconf_imprime_completa(fichero_matriz);
  if(verboso){
    printf("sust=%d ins=%d borr=%d ac=%d\n",ns,ni,nb,na);
    printf("p=%lf\n",p);
  }
  
  
  /*:28*/
//#line 1041 "tasas.w"
  
  if(con_matriz)mconf_free();
  /*8:*/
//#line 273 "tasas.w"
  
  free(d[0]);free(d);
  free(dd[0]);free(dd);
  free(va[0]);free(va);
  
  /*:8*/
//#line 1043 "tasas.w"
  
  libera_memoria_diccionario();
  exit(0);
}

/*:26*/