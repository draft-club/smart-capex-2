from IPython.display import display
import ipywidgets as widgets
from ipywidgets import Layout
import pandas as pd
import logging
import sys
from pathlib import Path
import os
from src import schema_tools
from notebooks import runner
from src.validation import Validation
from src.oss_validation import OssValidation
from ipywidgets import HBox, VBox
import io
import pandas as pd
import sys
import pandas as pd
from matplotlib import pyplot as plt
import logging
import pandas as pd
from IPython.display import clear_output
from copy import deepcopy

current_path = str(os.path.abspath(__file__))
path = Path(current_path)
root_dir = str(path.parent.parent.absolute())
sys.path.insert(0, root_dir)

sys.path.insert(0, '..')
logging.basicConfig(level=logging.INFO,
                    format='[ %(asctime)s %(levelname)-8s - %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    handlers=[logging.StreamHandler(sys.stdout)]
                    )

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 500)
# upload the file

df = None
schema = None
temp_schema = None
uploader = widgets.FileUpload(accept='.csv')
separator = widgets.Dropdown(options=[",", ";"], placeholder=",", description="separator",
                             layout=Layout(width='20%'))

output = widgets.Output()
output2 = widgets.Output()
output4 = widgets.Output()
text_init = widgets.Label('Upload data and create schema')
text_dataset_constraint = widgets.Label('Create dataset constraints')
text_column_constraint = widgets.Label('Add constraints for the columns')


# check if theres a file uploaded and create the dataframe
def upload_eventhandler(change):
    if len(uploader.get_state()['value']) >= 1:
        global df
        content = io.StringIO(str(uploader.value[0].content, 'utf8'))
        df = pd.read_csv(content, sep=separator.value)
        uploader.value = []


uploader.observe(upload_eventhandler, names='value')

# button to infer the first schema
button_infer = widgets.Button(value=False,
                              description='infer_schema')


# update schema if button is pressed
def button_eventhandler(change):
    # determine(choix2.label)
    # if button_infer.value:
    global schema
    schema = schema_tools.infer_schema(df)
    unicity_feat.options = df.columns
    # display(schema_tools.get_compact_schema(schema))


button_infer.on_click(button_eventhandler)

# button to display the schema
button_displaystuff = widgets.Button(value=False, description='display schema')


def displayevent(change):
    global schema
    if 'schema' in globals():
        with output:
            clear_output(wait=False)
            display(schema_tools.get_compact_schema(schema))


button_displaystuff.on_click(displayevent)

choix1 = widgets.Dropdown(description='choix colonne:', placeholder='choix colonne')
text_type = widgets.Dropdown(placeholder='truc', description='type', options=['str', 'int', 'float'])


def datapres_event(change):
    if 'df' in globals():
        choix1.options = df.columns
        join_left.options = df.columns
        # choix1.value = df.columns[1]

    # if 'schema' in globals():
    #    choix2.options = [('type', schema_tools.set_type), ('presence', schema_tools.set_presence),
    #                      ('domain', schema_tools.set_domain)]
    #    choix2.value = schema_tools.set_type


# def text_eventhandler(change):
#    #determine(choix2.label)
#    if choix2.label=='type':
#        arg.placeholder='infer type'
#    if choix2.label=='presence':
#        arg.placeholder='infer presence'

button_infer.on_click(datapres_event)
# choix2.observe(text_eventhandler, names='label')

conf_type = widgets.Button(description='validate constraint')


def confirmation_type(change):
    global temp_schema
    global schema
    temp_schema = deepcopy(schema)
    schema = schema_tools.set_type(schema, choix1.value, text_type.value)


conf_type.on_click(confirmation_type)

dropdown_display = widgets.Dropdown(description='display_constraint', placeholder='choix colonne')


def display_event(change):
    dropdown_display.options = schema_tools.get_compact_schema(schema)['element'].unique()


button_displaystuff.on_click(display_event)

button_refresh = widgets.Button(description='Refresh display')


def refresh_func(b):
    data = schema_tools.get_compact_schema(schema)
    data = data[data['element'] == dropdown_display.value]
    with output2:
        clear_output(wait=False)
        display(data)


button_refresh.on_click(refresh_func)

pres = widgets.BoundedFloatText(description='presence', min=0, max=1)

conf_presence = widgets.Button(value=False, description='validate constraint')


def confirmation_presence(change):
    global temp_schema
    global schema
    temp_schema = deepcopy(schema)
    schema = schema_tools.set_presence(schema, choix1.value, pres.value)


conf_presence.on_click(confirmation_presence)

min_dom = widgets.FloatText(description='minimum value')
max_dom = widgets.FloatText(description='maximum value')
dom = widgets.Text(description='possible values')

conf_domain = widgets.Button(description='validate constraint')


def confirmation_domain(change):
    global temp_schema
    global schema
    temp_schema = deepcopy(schema)
    schema = schema_tools.set_domain(schema=schema, min_value=min_dom.value, max_value=max_dom.value, domain=dom.value,
                                     feature_name=choix1.value)


conf_domain.on_click(confirmation_domain)

type_distinctness = widgets.Dropdown(description="type", options=["lt", "eq", "gt"])
distinctness_value = widgets.BoundedFloatText(min=0, max=1)

conf_distinctness = widgets.Button(description="validate constraint")


def confirmation_distinctness(change):
    global temp_schema
    global schema
    temp_schema = deepcopy(schema)
    schema = schema_tools.set_distinctness(schema=schema, feature_name=choix1.value,
                                           distinctness_condition=type_distinctness.value,
                                           distinctness_value=distinctness_value.value)


conf_distinctness.on_click(confirmation_distinctness)

tab_distinctness = VBox([type_distinctness, distinctness_value, conf_distinctness])

regex = widgets.Text(description="regex")
conf_regex = widgets.Button(description="validate constraint")


def confirmation_regex(change):
    global temp_schema
    global schema
    temp_schema = deepcopy(schema)
    schema = schema_tools.set_regex(schema=schema, feature_name=choix1.value, regex=regex.value)


conf_regex.on_click(confirmation_regex)
tab_regex = VBox([regex, conf_regex])

drift = widgets.FloatText(description="drift")
conf_drift = widgets.Button(description="validate constraint")


def confirmation_drift(change):
    global temp_schema
    global schema
    temp_schema = deepcopy(schema)
    schema = schema_tools.set_drift(schema=schema, feature_name=choix1.value, drift=drift.value)


conf_drift.on_click(confirmation_drift)
tab_drift = VBox([drift, conf_drift])

frequency = widgets.FloatText(description="high frequency")
conf_frequency = widgets.Button(description="validate constraint")


def confirmation_frequency(change):
    global temp_schema
    global schema
    temp_schema = deepcopy(schema)
    schema = schema_tools.set_highest_frequency_threshold(schema=schema, feature_name=choix1.value,
                                                          highest_frequency_threshold=drift.value)


conf_frequency.on_click(confirmation_drift)
tab_frequency = VBox([frequency, conf_frequency])

feature_to_be_mapped = widgets.Dropdown(description="mapped")
conf_map = widgets.Button(description="validate constraint")


def confirmation_mapping(change):
    global temp_schema
    global schema
    temp_schema = deepcopy(schema)
    schema = schema_tools.set_mapped_to_only_one(schema=schema, feature_name=choix1.value,
                                                 mapped_to_only_one=feature_to_be_mapped.value)


conf_map.on_click(confirmation_mapping)
tab_mapping = VBox([feature_to_be_mapped, conf_map])

# Dataset constraints

batch_size = widgets.BoundedFloatText(min=0)
conf_size = widgets.Button(description="batch size")


def confirmation_size(change):
    global temp_schema
    global schema
    temp_schema = deepcopy(schema)
    schema = schema = schema_tools.set_max_fraction_threshold(schema=schema,
                                                              threshold=batch_size.value)


conf_size.on_click(confirmation_size)
tab_size = VBox([batch_size, conf_size])

unicity_feat = widgets.SelectMultiple()
conf_unicity = widgets.Button(description="select unicity feature")


def confirmation_unicity(change):
    global temp_schema
    global schema
    temp_schema = deepcopy(schema)
    schema = schema_tools.set_unicity_features(schema=schema,
                                               features=list(unicity_feat.value))


conf_unicity.on_click(confirmation_unicity)
tab_unicity = VBox([unicity_feat, conf_unicity])

dataset_constraint_validate = widgets.Button(description='validate dataset')


def validate_dataset_constraint(change):
    data = schema_tools.get_compact_schema(schema)
    data = data[data['element'] == "dataset"]
    with output4:
        clear_output(wait=False)
        display(data)


dataset_constraint_validate.on_click(validate_dataset_constraint)


batch_id = widgets.Dropdown(description="select id column")
conf_id = widgets.Button(description="select batch id")


def confirmation_id(change):
    global temp_schema
    global schema
    temp_schema = deepcopy(schema)
    schema = schema_tools.set_batch_id_column(schema=schema,
                                              feature=batch_id.value)


conf_id.on_click(confirmation_id)
tab_id = VBox([batch_id, conf_id])

name_join = widgets.Text(description="name of join")
join_left = widgets.Dropdown()
join_right = widgets.Text(description="name of column")
join_threshold = widgets.FloatSlider(min=0, max=1, )
conf_join = widgets.Button(description="confirm join constraint")


def confirmation_join(change):
    global temp_schema
    global schema
    temp_schema = deepcopy(schema)
    schema = schema_tools.add_join_constraints(schema=schema,
                                               name=name_join.value,
                                               left_on=join_left.value,
                                               right_on=join_right.value,
                                               thresold=join_threshold.value)


conf_join.on_click(confirmation_join)
grid_join_1 = widgets.GridspecLayout(1, 20)
grid_join_1[0, 0:10] = join_left
grid_join_1[0, 10:20] = join_right

tab_join = VBox([name_join, grid_join_1, join_threshold, conf_join])

name_file = widgets.Text(placeholder='file name')
save_schema = widgets.Button(description='save schema')
save_class = widgets.Dropdown(options=["Validation", "OssValidation"])
validate_changes1 = widgets.Button(description='validate config')


def validate_config(change):
    file_type = name_file.value
    if save_class.value == "Validation":
        used_class = Validation
    else:
        used_class = OssValidation
    schema_tools.add_schema_to_config(schema, file_type)
    runner.add_type_class_matching(file_type, used_class)


validate_changes1.on_click(validate_config)

data_to_validate = widgets.FileUpload(accept='.csv')
n_examples = widgets.IntSlider(min=2, max=10, description='nb examples?')
validate_changes = widgets.Button(description='validate changes')
output3 = widgets.Output()


def validate_ch(change):
    global examples, report
    if len(data_to_validate.get_state()['value']) >= 1:
        content = data_to_validate.value[0]['content']
        content = io.StringIO(str(data_to_validate.value[0].content, 'utf8'))
        val_data = pd.read_csv(content)
    examples, report = runner.validate(name_file.value, data=val_data, n_examples=n_examples.value)
    with output3:
        clear_output(wait=False)
        display(report)


validate_changes.observe(validate_ch)

show_diff = widgets.Button(description='show diff')


def colors(x, l1, l2):
    l = []
    for i in x:
        if x['constraint'] in l1:
            l.append('background: red')
            l.append('background: red')
            l1.pop(0)
            break
        elif x['constraint'] in l2:
            l.append('background: green')
            l.append('background: green')
            l2.pop(0)
            break
        else:
            l.append('')
            l.append('')
            break
    return l


def show_changes_schema(schema1, schema2, elements):
    compact_schema_01 = schema_tools.get_compact_schema(schema1)
    compact_schema_1 = compact_schema_01[compact_schema_01['element'].isin(elements)]
    compact_schema_02 = schema_tools.get_compact_schema(schema2)
    compact_schema_2 = compact_schema_02[compact_schema_02['element'].isin(elements)]
    if len(compact_schema_1) == len(compact_schema_2):
        l = (compact_schema_1==compact_schema_2)['constraint']
        indexes = [x for x in l.index if l[x]==False]
        l1 = []
        l2 = []
        print(indexes)
        for i in indexes:
            l1.append(compact_schema_1.loc[i]['constraint'])
            l2.append(compact_schema_2.loc[i]['constraint'])
            compact_schema_1 = pd.concat([compact_schema_1.loc[:i],
                            pd.DataFrame(compact_schema_2.loc[i]).T,
                            compact_schema_1.loc[(i+1):]], axis=0)
            print(l1)
            print(l2)
        df = compact_schema_1.reset_index().drop('index', axis=1)
        return df.style.apply(lambda x: colors(x, l1, l2), axis=1)
    else:
        df = compact_schema_2.style.apply(lambda x: ['background: lightgreen' if x['constraint'] not in list(compact_schema_1['constraint']) else '' for i
                   in x], axis=1)
    #return df.style.apply(lambda x: ['background: #d65f5f' if x['constraint'] in l1 else 'background: lightgreen' if x['constraint'] in l2 else '' for i in x], axis=1)
        return df


def display_diff(change):
    new_df = show_changes_schema(temp_schema, schema, [choix1.value])
    with output2:
        clear_output(wait=False)
        display(new_df)


show_diff.on_click(display_diff)


tab_config = VBox(children=[name_file, save_schema, save_class, validate_changes1])
tab_validate = VBox(children=[data_to_validate, n_examples, validate_changes, output3])
tab_cool = widgets.Tab(children=[tab_config, tab_validate])
tab_cool.set_title(0, 'config')
tab_cool.set_title(1, 'validate')

tab1 = VBox(children=[separator, uploader, button_infer, button_displaystuff, output])
tab_type = VBox(children=[text_type, conf_type])
tab_presence = VBox(children=[pres, conf_presence])
tab_domain = VBox(children=[min_dom, max_dom, dom, conf_domain])

names = ['type', 'presence', 'domain', 'distinctness', 'regex', 'drift', 'high_freq', 'mapping']
tab3 = widgets.Tab(children=[tab_type, tab_presence, tab_domain, tab_distinctness, tab_regex, tab_drift,
                             tab_frequency, tab_mapping])
for i in range(8):
    tab3.set_title(i, names[i])

names_tab_dataset = ["batch size", "unicity column", "batch id", "join"]
tab_dataset = widgets.Tab(children=[tab_size, tab_unicity, tab_id, tab_join])
for i in range(4):
    tab_dataset.set_title(i, names_tab_dataset[i])

tab2 = VBox(children=[choix1, tab3, dropdown_display, button_refresh, show_diff, output2])
tab = widgets.Tab(children=[tab1, VBox([tab_dataset, dataset_constraint_validate, output4]), tab2, tab_cool])
tab.set_title(0, 'init')
tab.set_title(1, 'dataset constraint')
tab.set_title(2, 'column constraint')
tab.set_title(3, 'validate data')


def modify_entries(change):
    feature_to_be_mapped.options = tuple(x for x in choix1.options if x not in choix1.value)
    dropdown_display.value = choix1.value
    if tab3.get_title(tab3.selected_index) == 'domain':
        data = schema_tools.get_compact_schema(schema)
        l = data[data['element'] == choix1.value]['constraint']
        if any("type int" in k for k in l):
            min_dom.disabled = False
            max_dom.disabled = False
            dom.disabled = True
        elif any("type float" in k for k in l):
            min_dom.disabled = False
            max_dom.disabled = False
            dom.disabled = True
        else:
            dom.disabled = False
            max_dom.disabled = True
            min_dom.disabled = True


choix1.observe(modify_entries, names='value')
