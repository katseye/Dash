import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import base64
import io
import dash_table
import itertools

from flair.models import SequenceTagger
from flair.data import Sentence
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import polyglot
from polyglot.downloader import downloader
from polyglot.text import Text
from polyglot.detect import Detector
import math
import time

#from flask import Flask

model = SequenceTagger.load('ner')
downloader.download("TASK:transliteration2", quiet=True)

# Initialize the app
app = dash.Dash(__name__)
#application = app.server
#app = Flask(__name__)
app.config['suppress_callback_exceptions'] = True

app.layout = html.Div(
    children=[
        html.Div(className='row',
                 children=[
                    html.Div(className='three columns div-user-controls',
                             children=[
                                 html.H1('Automated Entity Recognition'),
                                 html.P('Classifying entities as person or organization'),
								 html.Hr(),
								 html.P('Step 1: Upload input file with entities and variants'),
								 html.P('Step 2: Select relevant entities from the displayed table'),
								 html.P('Step 3: Click button for automated entity recognition'),
								 html.Hr(),
                                 html.Div(
                                     className='div-for-upload',
                                     children=[
                                         dcc.Upload(id='upload-data', 
										 children=html.Div([
																html.Button('Upload File')
															]),
															style={
																	'width': '100%',
																	'height': '60px',
																	'lineHeight': '60px',
																	'borderWidth': '1px',
																	'borderStyle': 'dashed',
																	'borderRadius': '5px',
																	'textAlign': 'center',
																	'margin': '10px'
																},
																multiple=True
                                                      ),
                                     ],
                                     style={'color': '#1E1E1E'})
                                ]
                             ),
                    html.Div(className='nine columns div-for-charts bg-grey',
                             children=[
                                 html.Div(id='output-data-upload'),
								 html.Hr(),
                                 html.Div(id = 'output-ner')
                             ])
                    ])
        ]

)

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div(
	
	   children =  [
            dash_table.DataTable(
		    id = "datatable",
            data=df.to_dict('records'),
            columns=[{'name': i, 'id':i} for i in df.columns],
			editable=False,
			row_selectable="multi",
			row_deletable=False,
			selected_columns=[],
			selected_rows=[],
			page_action="native",
			page_current= 0,
			page_size= 8,
			style_table={
				'width':'940px','overflowX': 'auto',
				'height':'300px', 'overflowY': 'auto'
						},
			style_cell={
				'textAlign': 'left', 'color':'black',
				'height': 'auto',
				'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
				'whiteSpace': 'normal'
						
						},
        ),
        

        html.Hr(),  # horizontal line
        html.Div(id = 'ner_button_1',
			children = [
						html.Button(id = 'ner_button', children = 'Recognize Entities')
			],
			style={
				'width': '100%',
				'textAlign': 'center',
			}
		)
		#dcc.Store(id = 'memory')
		
        # For debugging, display the raw contents provided by the web browser
        #html.Div('Raw Content'),
        #html.Pre(contents[0:200] + '...', style={
        #    'whiteSpace': 'pre-wrap',
        #    'wordBreak': 'break-all'
        #})
    ])

def entity_recognition(text):
    #print('inside entity recognition')
    if isinstance(text, str):
        doc = text
    else:
        #print(type(text))
        doc = ''
        
    s = Sentence(doc.title())
    model.predict(s)
    #print('model')
    a = s.to_dict(tag_type = 'ner')
    #print(a)
    b = a['entities'][0]
    
    #print('inside entity recognition2')
    if len(b) > 0:
        #print('in if')
        #origText = b[0]['text']
        #print(b['labels'][0].to_dict())
        entity = b['labels'][0].to_dict()['value']
        #print(entity)
        confidence = round(b['labels'][0].to_dict()['confidence'],2)
        #print(confidence)
    else:
        #print('in else1')
        #origText = b[0]['text']
        entity = ''
        confidence = ''
    #print('inside entity recognition3')
    return entity, confidence

	
def transliteration(blob):
    text_Tr = ''
    if blob != '':
        text = Text(blob)
        
        try:
            text.transliterate('en')
        except ValueError:
            #print('in except')
            text_Tr = 'Language module not found'
        else:
            for j in range(len(text.transliterate('en'))):
                text_Tr = text_Tr+' '+text.transliterate('en')[j]
            text_Tr = text_Tr.strip()
    else:
        text_Tr = ''
    
    return text_Tr
	
def ner(rows_selected, row_indices):
    data=[rows_selected[i] for i in row_indices]
    #print('inside ner function: ', data)
    data = pd.DataFrame(data)
    #print('inside ner function2')
    #print(data.head())
	
	# Entity Recognition
    data['Type'], data['Conf'] = zip(*data['name'].map(entity_recognition))
    data['Type Alt1'], data['Conf Alt1'] = zip(*data['name version 1'].map(entity_recognition))
    data['Type Alt2'], data['Conf Alt2'] = zip(*data['name version 2'].map(entity_recognition))
    #print('after ner')
    #print(data.head())
	# Transliteration
    data['Transliteration'] = data['original language'].apply(str).apply(lambda x: transliteration(x))
    #print('after tln')
    #print(data.head())
    #print(data.columns)
	# Column cleanup
    data = data[['name', 'Type', 'Conf', 'name version 1', 'Type Alt1', 'Conf Alt1',
       'name version 2', 'Type Alt2', 'Conf Alt2', 'original language', 'Transliteration',
       'Actual']]
    #data_temp = data[['name', 'name version 1', 'name version 2', 'Transliteration']]
    #print('after ccp')
    #print(data_temp.head())   
    # Similarity Scoring
    data['Sim Alt1'] = data.apply(lambda x: fuzz.token_set_ratio(x['name'], x['name version 1']), axis=1)
    #print('after ssc1')
    data['Sim Alt2'] = data.apply(lambda x: fuzz.token_set_ratio(x['name'], x['name version 2']), axis=1)
    #print('after ssc2')
    data['Sim Orig Lang'] = data.apply(lambda x: fuzz.token_set_ratio(x['name'], x['Transliteration']), axis=1)
    #print('after ssc3')
    #print(data.head())
	# Consolidated Scoring
    data1 = data
    m1 = (data1[['Type', 'Type Alt1', 'Type Alt2']] == 'PER')
    m2 = (data1[['Type', 'Type Alt1', 'Type Alt2']] == 'ORG')
    scores = data1[['Conf', 'Conf Alt1', 'Conf Alt2']]
    data1['Per Score'] = pd.DataFrame(np.where(m1, scores, np.nan)).mean(skipna=True, axis=1).fillna(0)
    data1['Per Score'] = data1['Per Score'].apply(lambda x: round(x, 2))
    data1['Org Score'] = pd.DataFrame(np.where(m2, scores, np.nan)).mean(skipna=True, axis=1).fillna(0)
    data1['Org Score'] = data1['Org Score'].apply(lambda x: round(x, 2))
    #print('after cnsc')
    #print(data1.head())
	# Final tag
    col1 = 'Per Score'
    col2 = 'Org Score'
    conditions  = [ data1[col1] >= data1[col2], data1[col1] < data1[col2], data1[col1] == data1[col2] ]
    choices     = [ 'PER', 'ORG', 'UND' ]

    data1['Final Tag'] = np.where(data1[col1] >= data1[col2],'PER',np.where(data1[col1] < data1[col2],'ORG',
                                                                        np.where(data1[col1] == data1[col2],'UND','')))
    #print('after ftg')
    #print(data1.head())
    # Tag accuracy flagging
    data1['Tag Accuracy'] = (data1['Final Tag'] == data1.Actual).map({True:1, False:0})
    #print('after acfg')
    #print(data1.head())
	
    return html.Div([
	#html.P(filename),
	#html.H6(datetime.datetime.fromtimestamp(date)),

	dash_table.DataTable(
	    id = 'result_table',
		data = data1.to_dict('records'),
		columns=[{'name': i, 'id': i} for i in data1.columns],
		style_table={
			'width':'940px','overflowX': 'auto',
			'overflowY': 'auto'
					},
		style_cell={
			'textAlign': 'left', 'color':'black',
			'height': 'auto',
			'minWidth': '50px', 'width': '50px', 'maxWidth': '100px',
			'whiteSpace': 'normal'
					
					},
		editable=False,
		row_deletable=False,
		selected_columns=[],
		selected_rows=[],
		page_action="native",
		page_current= 0,
		page_size= 8,
		),
		
		html.Hr(),  # horizontal line
        html.Div(id = 'dl_button',
			children = [
		html.Button('Download')
			],
			style={
				'width': '100%',
				'textAlign': 'right',
			}
		),
		html.P(id = 'dl_button_hidden', style={'display':'none'})
	])
	   
    
	

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')]
			  )

def update_output(list_of_contents, list_of_names):
    print('Inside Update Output app.py')
    #print(list_of_names)
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in zip(list_of_contents, list_of_names)
			]
        #print(children)
        return children
		
# @app.callback(Output('memory', 'data'),
              # [Input('ner_button', 'n_clicks')],
			   # [State('memory', 'data')]
              # )
# def update_click_value(n_clicks, data):
    # print('In func update', n_clicks)
    # if n_clicks is None:
        # raise PreventUpdate
		
    # data = data or {'clicks': 0}
    # data['clicks'] = n_clicks-1
    # print('In func update 1', data.get('clicks'))
    # return data
	
@app.callback(Output('output-ner', 'children'),
              [Input('ner_button', 'n_clicks')],
			  [State('datatable', 'derived_virtual_data'),
              State('datatable', 'derived_virtual_selected_rows')
			  ]
			  )

def ner_output(n_clicks, rows_selected, row_indices):
    if n_clicks != None:
        print('Inside ner_output')
        print(n_clicks)
        #print(rows_selected)
        if rows_selected is not None:
            print('before ner func')
            children = [
                  ner(rows_selected, row_indices)
		    ]
            return children
        else:
            print(row_indices)
    
			
@app.callback(Output('dl_button_hidden', 'children'),
              [Input('dl_button', 'n_clicks'),
			  Input('result_table', 'derived_virtual_data')]
			  )

def dl_output(n_clicks, tableRows):
    if n_clicks != None:
        
        df = pd.DataFrame(tableRows)
        print(df)
        t2 = time.ctime().split()
        del t2[3]
        t2 = '_'.join(t2)
        name2 = 'Result_'	
        extn2 = '.xlsx'
        filename2 = t2.join([name2,extn2])
        path2 = 'C:/Anaconda3/envs/dash_env/App/results/'
        filepath2 = path2+filename2
        print(filepath2)
        df.to_excel(filepath2)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
