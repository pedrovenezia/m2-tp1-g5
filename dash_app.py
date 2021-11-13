import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from Calculadora_TL import Calculadora_TL

app = dash.Dash(__name__)
app.title = 'Calculadora TL'
server = app.server
calculadora_tl = Calculadora_TL('TABLA MATERIALES TP1.xlsx')
data = calculadora_tl.data
available_indicators = data.material.unique()

app.layout = html.Div([

    html.Div([html.Img(src=app.get_asset_url('untref.png'), style={'margin': 'auto'}),
              html.H1("Aislamiento de una pared monolítica", style={'float':'right',
                'font-family': 'verdana', 'width':'50%'})],
                style={'vertical-align': 'top'}),

        html.Div([
            html.P(id = 'text-material',children = 'Material'),
            dcc.Dropdown(
                id='material',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Acero'
            ),
            html.Div([
                html.P(id = 'text-dimensiones',children = 'Dimensiones (m)'),
                dcc.Input(
                    id='largo',
                    placeholder='Largo',
                    type='number',
                    value='Largo'
                    ),
                dcc.Input(
                    id='alto',
                    placeholder='Alto',
                    type='number',
                    value='Alto'),
                dcc.Input(
                    id='espesor',
                    placeholder='Espesor',
                    type='number',
                    value='Ancho')
            ]),      
        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
             html.Button('Exportar', id='boton_exportar'), dcc.Download(id="exportar"),
             dcc.Checklist(    
                 id='metodos',          
                options=[
                    {'label': 'Pared Simple', 'value': 'ley1'},
                    {'label': 'Modelo Sharp', 'value': 'sharp'},
                    {'label': 'ISO 12354-1', 'value': 'ISO'},
                    {'label': 'Modelo Davy', 'value': 'davy'}
                ],
                labelStyle={'display': 'block'},
                style={'marginBottom': 150, 'marginTop': 25}),
        ],style={'width': '30%', 'float': 'right', 'display': 'inline-block'}),

        dcc.Graph(id='indicator-graphic'),

        dcc.RadioItems(
            id='xaxis-type',
            options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
            value='Linear',
            labelStyle={'display': 'inline-block'})
])

@app.callback(
    Output('indicator-graphic', 'figure'),
    Input('material', 'value'),
    Input('alto', 'value'),
    Input('largo', 'value'),
    Input('espesor', 'value'),
    Input('metodos', 'value'),
    Input('xaxis-type', 'value'))
def update_graph(material,alto,largo,espesor,metodos, xaxis_type):

    calculadora_tl = Calculadora_TL(data_path='TABLA MATERIALES TP1.xlsx',
                                    t=espesor,
                                    l1=largo, l2=alto)
    resultados = calculadora_tl.calcular_r(material, metodos)
    resultados['frecuencia'] = calculadora_tl.f
    colors = px.colors.qualitative.Plotly
    fig = go.Figure()
    # Edit the layout
    fig.update_layout(title='Aislamiento a ruido aéreo según distintos métodos',
                   xaxis_title='Frecuencia (Hz)',
                   yaxis_title='R (dB)')
    fig.update_xaxes(type='linear' if xaxis_type == 'Linear' else 'log')
    names = {'ley1':'pared simple', 'sharp':'modelo Sharp', 'ISO':'ISO 12354-1', 'davy': 'modelo Davy'}
    for i, x in enumerate(metodos):
        fig.add_traces(go.Scatter(x=resultados['frecuencia'], y = resultados[x],
                                  mode = 'lines+markers', line=dict(color=colors[i]),
                                  name=names[x]))
    fig.show()
    return fig

@app.callback(
    Output("exportar", "data"),
    Input('boton_exportar', 'n_clicks'),
    Input('material', 'value'),
    Input('alto', 'value'),
    Input('largo', 'value'),
    Input('espesor', 'value'),
    Input('metodos', 'value'),
    prevent_initial_call=True,)
def download_func(boton_exportar, material, alto, largo, espesor, metodos):
    calculadora_tl = Calculadora_TL(data_path='TABLA MATERIALES TP1.xlsx',
                                    t=espesor,
                                    l1=largo, l2=alto)
    resultados = calculadora_tl.calcular_r(material, metodos)
    resultados['frecuencia'] = calculadora_tl.f
    resultados_df = pd.DataFrame(resultados)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'boton_exportar' in changed_id:
        return dcc.send_data_frame(resultados_df.to_excel, "resultados.xlsx", sheet_name="resultados")

if __name__ == '__main__':
    app.run_server(debug=True)