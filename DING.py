# Martin&Doumas_PLOS_bio simple simulation code. This code will allow a user to run versions of the simulations from the aforementioned paper. 
# 

# imports
import basicRunDORA_DING

parameters = {'asDORA': True, 'gamma': 0.3, 'delta': 0.1, 'eta': 0.9, 'HebbBias': 0.5,'bias_retrieval_analogs': True, 'use_relative_act': True, 'run_order': ['cdr', 'selectTokens', 'r', 'wp', 'm', 'p', 'f', 's', 'c'], 'run_cyles': 1000, 'write_on_iteration': 10, 'firingOrderRule': 'random', 'ignore_object_semantics': False, 'ignore_memory_semantics': True, 'exemplar_memory': False, 'recent_analog_bias': True, 'lateral_input_level': 5, 'screen_width': 1200, 'screen_height': 700, 'doGUI': True, 'GUI_update_rate': 1}

# load the sym file.
sym_file = open('testsim_DING.py', 'r')
sym_file.seek(0)  # to get to the beginning of the file.
symstring = ''
for line in sym_file:
    symstring += line
exec (symstring)
mysym = basicRunDORA_DING.buildNetwork_DING.interpretSymfile(symProps)
# make the DORA object.
memory = basicRunDORA_DING.dataTypes_DING.memorySet()
memory = basicRunDORA_DING.buildNetwork_DING.buildTheNetwork(mysym[0], memory)
# make the runDORA object.
network = basicRunDORA_DING.runDORA(memory, parameters)

# make the firing order. 
semantic_order = [['dry1', 'dry2', 'dry3'], ['fur1', 'fur2', 'fur3'], ['rubber1', 'rubber2', 'rubber3', 'rubbed1', 'rubbed2', 'rubbed3'], ['skin1', 'skin2', 'skin3']]
firing_order = []
for sem_set in semantic_order:
    word = []
    for semantic in network.memory.semantics:
        if semantic.name in sem_set:
            word.append(semantic)
    firing_order.append(word)

# run1.
network.initialize_run(mapping=False)
network.initialize_network_state()
units_dict = network.do_ding_ops(firing_order)
# write units_dict to file.
#json.dump(units_dict, open(file_name, 'w'))

# run2.
#network.do_retrieval()