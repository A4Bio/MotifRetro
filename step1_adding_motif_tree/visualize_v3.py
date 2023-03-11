import sys; sys.path.append("/gaozhangyang/experiments/MotifRetro")
from dash import Dash, dcc, html, Input, Output, ctx
import dash_cytoscape as cyto
from src.utils.paper_utils import get_ams_attach, smi2image, draw_smiles
import networkx as nx
import json
from networkx import json_graph
from networkx.readwrite import cytoscape_data
import base64
import dash

cyto.load_extra_layouts()

app = Dash(__name__)
server = app.server

# all_trees = json.load(open("/gaozhangyang/experiments/MotifRetro/data/uspto_50k/adding_motif_trees.json","r"))

img_name = "*OCc1ccccc1"
init_graph = {"*OCc1ccccc1":
    {"tree": {
          "element": [
               0,
               1
          ],
          "neighbor": [
               2
          ],
          "am_smi": "*[OH:1]",
          "smi": "*O",
          "freq": 168,
          "id": 13,
          "children": [
               {
                    "element": [
                         2,
                         3,
                         4,
                         5,
                         6,
                         7,
                         8
                    ],
                    "neighbor": [
                         1
                    ],
                    "am_smi": "[*:1][CH2:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1",
                    "smi": "*Cc1ccccc1",
                    "freq": 765,
                    "id": 12
               }
          ]
     }}
    }



def dict2imgdata(graph_data):
    key = list(graph_data.keys())[0]
    tree = json_graph.tree_graph(graph_data[key]['tree'])
    for idx in tree.nodes:
        node = tree.nodes[idx]
        node["img"] = smi2image(node['am_smi'])
    tree.add_node(-1, img = smi2image(key))
    cy_g = cytoscape_data(tree)
    return cy_g['elements']

canvas_style={'width': '100%', 'height': '400px'}

styles = {
    'output': {
        'overflow-y': 'scroll',
        'overflow-wrap': 'break-word',
        'height': 'calc(100% - 25px)',
        'border': 'thin lightgrey solid'
    },
    'tab': {'height': 'calc(98vh - 115px)'}
}

style_sheets = [{'style': {
                            'background-color': 'blue',
                            'shape' : 'rectangle',
                            'width':600,
                            'height':400,
                            'border-color': 'rgb(0,0,0)',
                            'border-opacity': 1.0,
                            'border-width': 0.0,
                            'color': '#4579e8',
                            'background-image':'data(img)',
                            'background-fit':'contain'},
                'selector': 'node'},
                {'style': {'width': 20.0,},'selector': 'edge'}]



app.layout = html.Div([
    
    html.Div([
        html.H3('Upload JSON Graph Data'),
        dcc.Upload(
            id='upload_json',
            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            }
        ),
        html.Div(id='output-graph', children=[
                                    cyto.Cytoscape(
                                        id='cytoscape-image-export',
                                        elements=dict2imgdata(init_graph), 
                                        stylesheet=style_sheets, 
                                        layout={'name':'breadthfirst'},
                                        style=canvas_style
                                    )])
    ], style={'width': '100%', 'display': 'inline-block'}),
    
    

    html.Div(className='four columns', children=[
        html.Div('Download graph:'),
        html.Button("as jpg", id="btn-get-jpg"),
        html.Button("as png", id="btn-get-png"),
        html.Button("as svg", id="btn-get-svg"),
    ]),
])



# 点击下载按钮-->调用cytoscape-image-export.generateImage下载图片
@app.callback(
    Output("cytoscape-image-export", "generateImage"),
    [
        Input("btn-get-jpg", "n_clicks"),
        Input("btn-get-png", "n_clicks"),
        Input("btn-get-svg", "n_clicks"),
    ])

def get_image(get_jpg_clicks, get_png_clicks, get_svg_clicks):
    ftype = "png"
    action = 'store'

    if ctx.triggered:
        if ctx.triggered_id != "tabs":
            action = "download"
            ftype = ctx.triggered_id.split("-")[-1]

    return {
        'type': ftype,
        'action': action,
        'filename':f"{img_name}.png"
        }


# 上传json-->更新output-graph.children的绘图数据
@app.callback(
    dash.dependencies.Output('output-graph', 'children'),
    [dash.dependencies.Input('upload_json', 'contents')])


def update_graph(contents):
    if contents is not None:
        contents = contents.split(',')[1]
        graph_data = json.loads(base64.b64decode(contents).decode('utf-8'))
        img_list = []
        for idx, (key, val) in enumerate(graph_data.items()):
            print(key)
            tree = json_graph.tree_graph(graph_data[key]['tree'])

            for idx in tree.nodes:
                node = tree.nodes[idx]
                node["img"] = smi2image(node['am_smi'])
            tree.add_node(-1, img = smi2image(key))
            cy_g = cytoscape_data(tree)
            
            img_list.append(cyto.Cytoscape(
                id='cytoscape-image-export',
                # id=f"img-{idx}",
                elements=cy_g["elements"], 
                stylesheet=style_sheets, 
                layout={'name':'breadthfirst'},
                style=canvas_style
            ))
            
            global img_name
            img_name = key
        print("OK")
        return img_list
    






if __name__ == "__main__":
    
    app.run_server(debug=True)
    print()



