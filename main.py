from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI()


@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes: str):
    '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes historicamente'''
    df = pd.read_csv('peli_mes.csv')
    d = df.loc[df.month == mes]
    d = d.original_title.to_list()
    return {'mes': mes, 'cantidad': d[0]}


@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia: str):
    '''Se ingresa el dia y la funcion retorna la cantidad de peliculas que se estrebaron ese dia historicamente'''
    df = pd.read_csv('peli_dia.csv')
    d = df.loc[df.day == dia]
    d = d.original_title.to_list()
    return {'dia': dia, 'cantidad': d[0]}


@app.get('/score_titulo/{titulo}')
def score_titulo(titulo: str):
    '''Se ingresa el título de una filmación esperando como respuesta el título, el año de estreno y el score'''
    df = pd.read_csv('pelipop[.csv')
    d = df.loc[df.title == titulo]
    c = d['popularity'].to_list()[0]
    m = d['year'].to_list()[0]

    return {'titulo': titulo, 'popularidad': c, 'anio': m}


@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo: str):
    '''Se ingresa el título de una filmación esperando como respuesta el título, la cantidad de votos y el valor promedio de las votaciones.
    La misma variable deberá de contar con al menos 2000 valoraciones,
    caso contrario, debemos contar con un mensaje avisando que no cumple esta condición y que por ende, no se devuelve ningun valor.'''
    df = pd.read_csv('pelivot.csv')
    d = df.loc[df.title == titulo]
    c = d['vote_average'].to_list()[0]
    a = d['vote_count'].to_list()[0]
    m = d['year'].to_list()[0]
    return {'titulo': titulo, 'anio': m, 'voto_total': a, 'voto_promedio': c}





@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor: str):
    '''Se ingresa el nombre de un actor que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno.
        Además, la cantidad de películas que en las que ha participado y el promedio de retorno'''
    df = pd.read_csv('actor.csv')
    df1 = pd.read_csv('act_pel.csv')
    a = df.loc[df.actor == nombre_actor]
    r = a['return'].to_list()[0]
    c = df1[df1['actor'] == nombre_actor].shape[0]

    return {'actor':nombre_actor, 'cantidad_filmaciones':c, 'retorno_promedio':r}




@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    ''' Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno.
    Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.'''
    df = pd.read_csv('dir.csv')
    df1 = pd.read_csv('dir_pel.csv')
    d = df.loc[df.director == nombre_director]
    r = d['return'].to_list()

    b = df1[(df1['director'] == nombre_director) & (df1['return'].notnull())]
    g = b['title'].to_list()
    a = b['year'].to_list()
    rr = b['return'].to_list()
    bd = b['budget'].to_list()
    rv = b['revenue'].to_list()
    return {'director':nombre_director, 'retorno_total_director':r,
    'peliculas':g, 'anio':a, 'retorno_pelicula':rr,
    'budget_pelicula':bd, 'revenue_pelicula':rv}


@app.get('/recomendacion/{titulo}')
def recomendacion(titulo: str):
    '''Ingresasun nombre de pelicula y te recomienda las similares en una lista'''
    i = pd.read_csv("titulos.csv").iloc[:5000]
    tfidf = TfidfVectorizer(stop_words="english")
    i["overview"] = i["overview"].fillna("")

    tfidf_matriz = tfidf.fit_transform(i["overview"])
    coseno_sim = linear_kernel(tfidf_matriz, tfidf_matriz)

    indices = pd.Series(i.index, index=i["title"]).drop_duplicates()
    idx = indices[titulo]
    simil = list(enumerate(coseno_sim[idx]))
    simil = sorted(simil, key=lambda x: x[1], reverse=True)
    simil = simil[1:11]
    movie_index = [i[0] for i in simil]

    lista = i["title"].iloc[movie_index].to_list()[:5]

    return {'lista recomendada': lista}
