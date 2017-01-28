# basicRunDORA.py

# basic functions and core run code for a simple DORA run (i.e., mapping, retrieval, predication, schema induction, whole relation formation).

# Are you being run on an iPhone?
run_on_iphone = False

# imports.
import random, numbers, math, operator
import numpy as np
import dataTypes_DING
import buildNetwork_DING
import DORA_GUI_ding
if not run_on_iphone:
    import pygame
    from pygame.locals import *
import pdb

# Initialize pygame screen size to 1200x800.
screen_width = 1200.0
screen_height = 700.0

# class that performs all the run operations in DORA. In class form so that new operations (e.g., compression, predicate recognition) can be implemented as new functions in this class (under the phase set section).
class runDORA(object):
    def __init__(self, memory, parameters):
        self.memory = memory
        self.firingOrderRule = parameters['firingOrderRule']
        self.firingOrder = None # initialized to None.
        self.asDORA = parameters['asDORA']
        self.gamma = parameters['gamma']
        self.delta = parameters['delta']
        self.eta = parameters['eta']
        self.HebbBias = parameters['HebbBias']
        self.bias_retrieval_analogs = parameters['bias_retrieval_analogs']
        self.use_relative_act = parameters['use_relative_act']
        if run_on_iphone:
            self.doGUI = False
        else:
            self.doGUI = parameters['doGUI']
        self.screen = 0
        self.GUI_information = None # initialize to None.
        self.screen_width = parameters['screen_width']
        self.screen_height = parameters['screen_height']
        self.GUI_update_rate = parameters['GUI_update_rate']
        self.ignore_object_semantics = parameters['ignore_object_semantics']
        self.ignore_memory_semantics = parameters['ignore_memory_semantics']
        self.exemplar_memory = parameters['exemplar_memory']
        self.recent_analog_bias = parameters['recent_analog_bias']
        self.lateral_input_level = parameters['lateral_input_level']
        self.num_phase_sets_to_run = None
        self.count_by_RBs = None # initialize to None.
        self.local_inhibitor_fired = False # initialize to False.
    
    ######################################
    ###### DORA OPERATION FUNCTIONS ######
    ######################################
    # 1) Bring a prop or props into WM (driver). This step is completed by passing the variable memory as an argument to the function (memory contains the driver proposition(s)).
    # function to prepare runDORA object for a run.
    def initialize_run(self, mapping):
        # index memory.
        self.memory = indexMemory(self.memory)
        # set up driver and recipient.
        self.memory.driver.Groups = []
        self.memory.driver.Ps = []
        self.memory.driver.RBs = []
        self.memory.driver.POs = []
        self.memory.recipient.Groups = []
        self.memory.recipient.Ps = []
        self.memory.recipient.RBs = []
        self.memory.recipient.POs = []
        if mapping == True and self.exemplar_memory == True:
            self.memory = make_AM_copy(self.memory)
        else:
            self.memory = make_AM(self.memory)
        # initialize .same_RB_POs field for POs.
        self.memory = update_same_RB_POs(self.memory)
        # initialize GUI if necessary.
        if self.doGUI:
            self.screen, self.GUI_information = DORA_GUI_ding.initialize_GUI(self.screen_width, self.screen_height, self.memory)
        # get PO SemNormalizations.
        for myPO in self.memory.POs:
            myPO.get_weight_length()
    
    # 2) Initialize activations and inputs of all units to 0.
    def initialize_network_state(self):
        self.memory = initialize_memorySet(self.memory)
        self.inferred_new_P = False
    
    # 3) Select firing order of RBs in the driver (for now this step is random or user determined).
    def create_firing_order(self):
        if len(self.memory.driver.RBs) > 0:
            self.count_by_RBs = True
        else:
            self.count_by_RBs = False
            # and randomly assign the PO firing order.
            self.firingOrder = []
            for myPO in self.memory.driver.POs:
                self.firingOrder.append(myPO)
            random.shuffle(self.firingOrder)
        if self.count_by_RBs:
            self.firingOrder = makeFiringOrder(self.memory, self.firingOrderRule)
    
    # function to perform steps 1-3 above.
    def do_1_to_3(self, mapping):
        self.initialize_run(mapping)
        self.initialize_network_state()
        self.create_firing_order()
    
    # 4) Enter the phase set. A phase set is each RB firing at least once (i.e., all RBs in firingOrder firing). It is in phase_sets you will do all of DORA's interesting operations (retrieval, mapping, learning, etc.). There is a function for each interesting operation.
    def do_retrieval(self):
        # do initialize network operations (steps 1-3 above). 
        self.do_1_to_3(mapping=False)
        phase_sets = 1
        for phase_set in range(phase_sets):
            LTM_list = []
            # fire all RBs in self.firingOrder, unless there are no RBs, in which case fire the POs in self.firingOrder. 
            if len(self.memory.driver.RBs) > 0:
                for currentRB in self.firingOrder:
                    # initialize phase_set_iterator and flags (local_inhibitor_fired).
                    phase_set_iterator = 1
                    self.local_inhibitor_fired = False
                    # 4.1-4.2) Fire the current RB in the firingOrder. Update the network in discrete time-steps until the globalInhibitor fires (i.e., the current active RB is inhibited by its inhibitor).
                    while self.memory.globalInhibitor.act == 0:
                        # 4.3.1-4.3.10) update network activations.
                        currentRB.act = 1.0
                        self.time_step_activations(phase_set, self.ignore_object_semantics, self.ignore_memory_semantics)
                        # 4.3.12) Run retrieval routines.
                        self.memory = retrieval_routine(self.memory, self.asDORA, self.gamma, self.delta, self.HebbBias, self.lateral_input_level, self.bias_retrieval_analogs)
                        # fire the local_inhibitor if necessary.
                        self.time_step_fire_local_inhibitor()
                        # update GUI.
                        phase_set_iterator += 1
                        if self.doGUI:
                            self.time_step_doGUI(phase_set_iterator)
                        # write state of LTM to a vector.
                        LTM_vec = []
                        for myP in self.memory.Ps:
                            LTM_vec.append(myP.act)
                        for myRB in self.memory.RBs:
                            LTM_vec.append(myRB.act)
                        for myPO in self.memory.POs:
                            LTM_vec.append(myPO.act)
                        LTM_list.append(LTM_vec)
                    # RB firing is OVER.
                    self.post_count_by_operations()
            else:
                # when you are retrieving by POs, you are firing the POs one at a time in the driver by default as the firing order is composed of POs only. As a consequence, when you are running in LISA mode and PO inhibitors are not updating, you will not get PO time sharing (i.e., a PO will keep firing forever, as it's inhibitor is not updating (in LISA mode PO inhibitors do not update)). So, you must move to DORA mode for retrieval.
                # set .asDORA to True.
                previous_mode = self.asDORA
                self.asDORA = True
                for currentPO in self.firingOrder:
                    # initialize phase_set_iterator and flags (local_inhibitor_fired).
                    phase_set_iterator = 1
                    self.local_inhibitor_fired = False
                    # 4.1-4.2) Fire the current RB in the firingOrder. Update the network in discrete time-steps until the globalInhibitor fires (i.e., the current active RB is inhibited by its inhibitor).
                    while self.memory.localInhibitor.act == 0:
                        # 4.3.1-4.3.10) update network activations.
                        currentPO.act = 1.0
                        self.time_step_activations(phase_set, self.ignore_object_semantics, self.ignore_memory_semantics)
                        # 4.3.12) Run retrieval routines.
                        self.memory = retrieval_routine(self.memory, self.asDORA, self.gamma, self.delta, self.HebbBias, self.lateral_input_level, self.bias_retrieval_analogs)
                        # fire the local_inhibitor if necessary.
                        self.time_step_fire_local_inhibitor()
                        # update GUI.
                        phase_set_iterator += 1
                        if self.doGUI:
                            self.time_step_doGUI(phase_set_iterator)
                    # PO firing is OVER.
                    self.post_count_by_operations()
                # return the .asDORA setting to its previous state.
                self.asDORA = previous_mode    
            # phase set is OVER.
            pdb.set_trace()
            self.post_phase_set_operations(retrieval_license=True, map_license=False)
    
    # function to do any weird test-type operations that you want to play around with. Some might be appropriated for the model's actual operation later.
    def do_ding_ops(self, firing_order):
        # do sentence processing stuff as in the Ding et al. (2016) paper. 
        # crux is that RBs can take children, and you're firing by semantics in a specific order. 
        # set .asDORA to True.
        previous_mode = self.asDORA
        self.asDORA = True
        self.count_by_RBs = True
        # make a dict that includes all units in the network with a list for each unit where the activation of that unit for all iterations can be stored.
        units_dict = {}
        for myP in self.memory.Ps:
            units_dict[myP.name] = []
        for myRB in self.memory.RBs:
            units_dict[myRB.name] = []
        for myPO in self.memory.POs:
            units_dict[myPO.name] = []
        for mysemantic in self.memory.semantics:
            units_dict[mysemantic.name] = []
        # fire the word list. 
        for pattern in firing_order:
            # initialize phase_set_iterator and flags (local_inhibitor_fired).
            phase_set_iterator = 1
            self.local_inhibitor_fired = False
            # 4.1-4.2) Fire the current RB in the firingOrder. Update the network in discrete time-steps until the globalInhibitor fires (i.e., the current active RB is inhibited by its inhibitor).
            while phase_set_iterator <= 110:
                # set activation of active semantic units to 1.
                for semantic in pattern:
                    semantic.act = 1.0
                # update the RBmodes.
                for myRB in self.memory.driver.RBs:
                    myRB.get_RBmode()
                for myRB in self.memory.recipient.RBs:
                    myRB.get_RBmode()
                # 4.3.1-4.3.10) update network activations.
                self.time_step_activations(1, self.ignore_object_semantics, self.ignore_memory_semantics, True)
                # fire the local_inhibitor if necessary.
                self.time_step_fire_local_inhibitor()
                # add each units activation to units_dict.
                for myP in self.memory.Ps:
                    units_dict[myP.name].append(myP.act)
                for myRB in self.memory.RBs:
                    units_dict[myRB.name].append(myRB.act)
                for myPO in self.memory.POs:
                    units_dict[myPO.name].append(myPO.act)
                for mysemantic in self.memory.semantics:
                    units_dict[mysemantic.name].append(mysemantic.act)
                # update the phase_set_iterator.
                phase_set_iterator += 1
                # GUI.
                self.time_step_doGUI(phase_set_iterator)
            # pattern/word firing is OVER.
            # fire the globalInhibitor.
            self.memory = self.memory.globalInhibitor.fire_global_inhibitor(self.memory)
            # reset the memory.localInhibitor.act and memory.globalInhibitor.act back to 0.0.
            self.memory.localInhibitor.act = 0.0
            self.memory.globalInhibitor.act = 0.0
            for myPO in self.memory.POs:
                myPO.reset_inhibitor()
            for semantic in self.memory.semantics:
                semantic.act = 0.0
        # return the .asDORA setting to its previous state.
        self.asDORA = previous_mode
        self.count_by_RBs = None
        # phase set is OVER.
        self.post_phase_set_operations(retrieval_license=False, map_license=False) 
        # return the units_dict.
        return units_dict
    
    
    ######################################################################
    ######################################################################
    ######################################################################
    
    ######################################
    ###### DORA TIME_STEP FUNCTIONS ######
    ######################################
    # functions implementing operations performed during a single time-step in DORA.
    # function to perform basic network activation update for a time_step in the phase set.
    def time_step_activations(self, phase_set, ignore_object_semantics=False, ignore_memory_semantics=False, do_ding=False):
        # initialize the input to all tokens and semantic units.
        self.memory = initialize_input(self.memory)
        # 4.3.2) Update modes of all P units in the driver and the recipient.
        if self.count_by_RBs:
            for myP in self.memory.driver.Ps:
                myP.get_Pmode()
            for myP in self.memory.recipient.Ps:
                myP.get_Pmode()
        # 4.3.3) Update input to driver token units.
        self.memory = update_driver_inputs(self.memory, self.asDORA, self.lateral_input_level)
        # 4.3.4-5) Update input to and activation of PO and RB inhibitors.
        for myRB in self.memory.driver.RBs:
            myRB.update_inhibitor_input()
            myRB.update_inhibitor_act()
        # update PO inhibitor act only if in DORA mode (i.e., asDORA == True).
        for myPO in self.memory.driver.POs:
            myPO.update_inhibitor_input()
            if self.asDORA:
                myPO.update_inhibitor_act()
        for myRB in self.memory.recipient.RBs:
            myRB.update_inhibitor_input()
            myRB.update_inhibitor_act()
        for myPO in self.memory.recipient.POs:
            myPO.update_inhibitor_input()
            if self.asDORA:
                myPO.update_inhibitor_act()
        # 4.3.6-7) Update input and activation of local and global inhibitors.
        self.memory.localInhibitor.checkDriverPOs(self.memory)
        self.memory.globalInhibitor.checkDriverRBs(self.memory)
        # 4.3.8) Update input to semantic units, unless you are running a Ding sim.
        for semantic in self.memory.semantics:
            # ignore input to semantic units from POs in object mode if ignore_object_semantics==True (i.e., if DORA is focusing on relational properties (from Hummel & Holyoak, 2003)).
            semantic.update_input(self.memory, ignore_object_semantics, ignore_memory_semantics)
        # 4.3.9) Update input to all tokens in the recipient and emerging recipient (i.e., newSet).
        self.memory = update_recipient_inputs(self.memory, self.asDORA, phase_set, self.lateral_input_level, self.ignore_object_semantics)
        self.memory = update_newSet_inputs(self.memory)
        # 4.3.10) Update activations of all units in the driver, recipient, and newSet, and all semanticss.
        self.memory = update_activations_run(self.memory, self.gamma, self.delta, self.HebbBias, phase_set, do_ding)
    
    # function to fire the local inhibitor if necessary.
    def time_step_fire_local_inhibitor(self):
        if self.asDORA and self.memory.localInhibitor.act >= 0.99 and not self.local_inhibitor_fired:
            self.memory = self.memory.localInhibitor.fire_local_inhibitor(self.memory)
            self.local_inhibitor_fired = True
    
    # function to do GUI.
    def time_step_doGUI(self, phase_set_iterator):
        if self.doGUI:
            # check for keypress for pause.
            debug = False
            pause = False
            for event in pygame.event.get():
                if not hasattr(event,'key'):
                    continue
                elif event.key == K_p and event.type == KEYDOWN:
                    # graphics are paused, wait for unpause.
                    pause = True
            if pause:
                wait = True
                while wait:
                    for event2 in pygame.event.get():
                        if event2.type == KEYDOWN:
                            if event2.key == K_p:
                                pause = False
                                wait = False
                            elif event2.key == K_d:
                                # enter debug.
                                debug = True
            ###########################################################
            ############### ENTER DEBUGGING DURING RUN! ###############
            # NOTE: for DEBUGGING, enters set_trace() after a GUI pause.
            ###########################################################
            if debug:
                pdb.set_trace()
            # check for update GUI.
            if phase_set_iterator % self.GUI_update_rate == 0:
                # update_GUI.
                self.screen, self.memory = DORA_GUI_ding.run_GUI(self.screen, self.GUI_information, self.memory, False)
    
    ####################################################
    #### DORA POST COUNT_BY AND PHASE_SET FUNCTIONS ####
    ####################################################
    # function to perform operations that occur after PO (if firing by POs) or RB (if firing by RBs) fires (i.e., what we're calling "count_by" operations as they occur after the firing of of the token you're firing (or counting) by).
    def post_count_by_operations(self):
        # fire the globalInhibitor.
        self.memory = self.memory.globalInhibitor.fire_global_inhibitor(self.memory)
        # reset the memory.localInhibitor.act and memory.globalInhibitor.act back to 0.0.
        self.memory.localInhibitor.act = 0.0
        self.memory.globalInhibitor.act = 0.0
        # reset the RB and PO inhibitors.
        for myRB in self.memory.RBs:
            myRB.reset_inhibitor()
        for myPO in self.memory.POs:
            myPO.reset_inhibitor()
    
    # functions to perform post-phase_set operations. 
    def post_phase_set_operations(self, retrieval_license, map_license, inferred_new_P=False):
        # if you were doing retrieval (i.e., if retrieval_license is True), then use the Luce choice axiom here to retrieve items from memorySet into the recipient.
        if retrieval_license:
            self.memory = retrieve_tokens(self.memory, self.bias_retrieval_analogs, self.use_relative_act)
        # reset the mode of all P units in the recipient back to neutral (i.e., 0);
        for myP in self.memory.recipient.Ps:
            myP.initialize_Pmode()
        # reset the RB and PO inhibitors.
        for myRB in self.memory.RBs:
            myRB.reset_inhibitor()
        for myPO in self.memory.POs:
            myPO.reset_inhibitor()
        # reset the activation and input of all units back to 0.
        self.memory = initialize_AM(self.memory)

######################################################################

#######################################
######### CORE DORA FUNCTIONS #########
#######################################
# function to make AM without making copies of items in LTM.
# noinspection PyPep8Naming
def make_AM(memory):
    # for each token, if it is in an AM set (driver, recipient), then make sure all sub-tokens are also in that set. Make sure that all tokens from the same analog are in the same set or in memory (i.e., tokens from the same analog CANNOT be in different AM sets). run findDriverRecipient().
    # for each token unit, make sure all subtokens are in the same set. Also, make sure that if a token is to enter recipient, that it checks to make sure none of it's subtokens are in the driver, and, if they are, that it remains in memory.
    for Group in memory.Groups:
        if Group.set != 'memory':
            # make sure all subtokens are in the same set.
            Group = set_sub_tokens(Group)
    for myP in memory.Ps:
        if myP.set != 'memory':
            # make sure all subtokens are in the same set.
            myP = set_sub_tokens(myP)
    for myRB in memory.RBs:
        if myRB.set != 'memory':
            # make sure all subtokens are in the same set.
            myRB = set_sub_tokens(myRB)
    for myPO in memory.POs:
        if myPO.set != 'memory':
            # make sure all subtokens are in the same set.
            myPO = set_sub_tokens(myPO)
    # and bring all the copied items from memory into AM (i.e., into driver/recipient).
    memory = findDriverRecipient(memory)
    # done.
    return memory

# function to make sure all sub-tokens of a token to enter AM are in the same set. Function also checks that any item to enter the recipient, does not have driver sub-tokens.
def set_sub_tokens(token):
    # check what kind of token you're dealing with.
    if token.my_type == 'Group':
        # if you're dealing with a Group, then for each of it's sub-groups, sub-Ps, and sub-RBs, set that sub_token.set to the same set as the Group, and run set_sub_tokens on that sub-token.
        # check to make sure that if the Group is in the recipient, none of it's subtokens are in the driver.
        go_on = True
        if token.set == 'recipient':
            go_on = check_sub_tokens(token)
        if go_on:
            for Group_under in token.myChildGroups:
                Group_under.set = token.set
                Group_under = set_sub_tokens(Group_under)
            for myP in token.myPs:
                myP.set = token.set
                myP = set_sub_tokens(myP)
            for myRB in token.myRBs:
                myRB.set = token.set
                myRB = set_sub_tokens(myRB)
        else:
            # set the token.set to 'memory'.
            token.set = 'memory'
    elif token.my_type == 'P':
        # if you're dealing with a P, then for each of it's RBs, set that myRB.set to the same set as the P, and run set_sub_tokens on the myRB.
        # check to make sure that if the P is in the recipient, none of it's subtokens are in the driver.
        go_on = True
        if token.set == 'recipient':
            go_on = check_sub_tokens(token)
        if go_on:
            for myRB in token.myRBs:
                myRB.set = token.set
                myRB = set_sub_tokens(myRB)
        else:
            # set the token.set to 'memory'.
            token.set = 'memory'
    elif token.my_type == 'RB':
        # if you're dealing with a RB, then for each of it's child-Ps and POs, set that token.set to the same set as the RB, and run set_sub_tokens on the the sub-token.
        # check to make sure that if the RB is in the recipient, none of it's subtokens are in the driver.
        go_on = True
        if token.set == 'recipient':
            go_on = check_sub_tokens(token)
        if go_on:
            if len(token.myPred) > 0:
                token.myPred[0].set = token.set
                token.myPred[0] = set_sub_tokens(token.myPred[0])
            if len(token.myObj) > 0:
                token.myObj[0].set = token.set
                token.myObj[0] = set_sub_tokens(token.myObj[0])
            elif len(token.myChildP) > 0:
                token.myChildP[0].set = token.set
                token.myChildP[0] = set_sub_tokens(token.myChildP[0])
        else:
            # set the token.set to 'memory'.
            token.set = 'memory'
    # done.
    return token

# function to check all sub-tokens of a token bound for the recipient, to make sure that none are in the driver.
def check_sub_tokens(token):
    # set the go_on_flag to True (indicating that that there are no driver sub-tokens of a recipient super-token).
    go_on_flag = True
    # make sure you're dealing with a recipient token (this is a redundent check, but is here for safety).
    if token.set == 'recipient':
        # make sure all sub-tokens are NOT in the driver.
        if token.my_type == 'Group':
            # make sure none of my sub-groups (or their sub-tokens) are in the driver.
            for sub_group in token.myGroups:
                if sub_group.set == 'driver':
                    # set token.set to 'memory', go_on_flag to False, and break the loop.
                    token.set = 'memory'
                    go_on_flag = False
                    break
                else:
                    # make sure none of the sub-token of the sub-group are in the driver.
                    go_on_flag = check_sub_tokens(sub_group)
                    # if go_on_flag is now False (i.e., the token has sub-tokens in the driver), token.set should be set to 'memory' (as it should not be retrieved into the recipient). 
                    if not go_on_flag:
                        token.set == 'memory'
            # if go_on_flag is still True, make sure none of my Ps (or their sub-tokens) are in the driver.
            if go_on_flag:
                for myP in token.myPs:
                    if myP.set == 'driver':
                        # set token.set to 'memory', go_on_flag to False, and break the loop.
                        token.set = 'memory'
                        go_on_flag = False
                        break
                    else:
                        # make sure none of the sub-token of the sub-group are in the driver.
                        go_on_flag = check_sub_tokens(myP)
                        # if go_on_flag is now False (i.e., the token has sub-tokens in the driver), token.set should be set to 'memory' (as it should not be retrieved into the recipient). 
                        if not go_on_flag:
                            token.set == 'memory'
            # if go_on_flag is still True, make sure none of my RBs (or their sub-tokens) are in the driver.
            if go_on_flag:
                for myRB in token.myRBs:
                    if myRB.set == 'driver':
                        # set token.set to 'memory', go_on_flag to False, and break the loop.
                        token.set = 'memory'
                        go_on_flag = False
                        break
                    else:
                        # make sure none of the sub-token of the sub-group are in the driver.
                        go_on_flag = check_sub_tokens(myRB)
                        # if go_on_flag is now False (i.e., the token has sub-tokens in the driver), token.set should be set to 'memory' (as it should not be retrieved into the recipient). 
                        if not go_on_flag:
                            token.set == 'memory'
        elif token.my_type == 'P':
            # make sure none of my RBs (or their sub-tokens) are in the driver.
            for myRB in token.myRBs:
                if myRB.set == 'driver':
                    # set token.set to 'memory', go_on_flag to False, and break the loop.
                    token.set = 'memory'
                    go_on_flag = False
                    break
                else:
                    # make sure none of the sub-token of the sub-group are in the driver.
                    go_on_flag = check_sub_tokens(myRB)
                    # if go_on_flag is now False (i.e., the token has sub-tokens in the driver), token.set should be set to 'memory' (as it should not be retrieved into the recipient). 
                    if not go_on_flag:
                        token.set == 'memory'
        elif token.my_type == 'RB':
            # make sure none of my POs are in the driver.
            if token.myPred[0].set == 'driver':
                # set token.set to 'memory' and go_on_flag to False.
                token.set == 'memory'
                go_on_flag = False
            # make sure you only check the objs of RBs not taking a child P as an argument.
            if len(token.myObj) > 0:
                if token.myObj[0].set == 'driver':
                    # set token.set to 'memory' and go_on_flag to False.
                    token.set == 'memory'
                    go_on_flag = False
    # done.
    return go_on_flag

# function to make copies of items from memory to enter AM.
def make_AM_copy(memory):
    # go through memory and make a list of all analogs to be copied. For each item, if it is to be retrieved into AM, then check if its analog is in the list of analogs to enter AM. If not, add it.
    analogs_to_copy = []
    for analog in memory.analogs:
        # check if the analog is to be copied, and if so, copy it.
        copy_analog_flag = check_analog_for_tokens_to_copy(analog)
        if copy_analog_flag and (analog not in analogs_to_copy):
            analogs_to_copy.append(analog)
    # now copy all analogs from analogs_to_copy into memory.
    for analog in analogs_to_copy:
        memory = copy_analog(analog, memory)
    # and bring all the copied items from memory into AM (i.e., into driver/recipient).
    memory = findDriverRecipient(memory)
    # all done.
    return memory

# function to check an analog for whether it contains any tokens to copy.
def check_analog_for_tokens_to_copy(analog):
    # check if analog is to be copied. Go through all tokens in the analog and see whether the .set field of any is NOT 'memory'. If it is not, break the loop and copy the analog to analogs_to_copy.
    copy_analog_flag = False
    # first check all the Groups.
    if not copy_analog_flag:
        for Group in analog.myGroups:
            if Group.set != 'memory':
                # set copy_analog_flag to True and break the loop.
                copy_analog_flag = True
                break
    # if copy_analog_flag is still False, check the Ps.
    if not copy_analog_flag:
        for myP in analog.myPs:
            if myP.set != 'memory':
                # set copy_analog_flag to True and break the loop.
                copy_analog_flag = True
                break
    # if copy_analog_flag is still False, check the RBs.
    if not copy_analog_flag:
        for myRB in analog.myRBs:
            if myRB.set != 'memory':
                # set copy_analog_flag to True and break the loop.
                copy_analog_flag = True
                break
    # if copy_analog_flag is still False, check the POs.
    if not copy_analog_flag:
        for myPO in analog.myPOs:
            if myPO.set != 'memory':
                # set copy_analog_flag to True and break the loop.
                copy_analog_flag = True
                break
    # return copy_analog_flag
    return copy_analog_flag

# function to copy a to be retrieved analog and it's elements into AM.
def copy_analog(analog, memory):
    # make a copy of the analog. NOTE: you can't use copy here because of recursion issues, so you're rolling your own copy code. Maybe there's a package for this, but then you're you, so you're not looking it up.
    new_analog = dataTypes.Analog()
    # make all tokens from the to be copied analog. 
    new_analog, memory = copy_analog_tokens(analog, new_analog, memory)
    # in the original analog, set the .set field of each element to 'memory'. 
    analog = clear_set(analog)
    # for each token in the copied analog, if a token is to be retrieved, then make sure all tokens below it are also to be retrieved (e.g., if a P is to be retrieved into 'recipient', make sure all RBs and POs connected to those RBs also have their .set field set to 'recipient).
    new_analog = retrieve_all_relevant_tokens(new_analog)
    # for each token in the copied analog, delete any token that is not be be retrieved (i.e., the .set field is 'memory') (I don't think you need this part: AND there are no higher tokens that are to be retrieved (e.g., a PO has no RBs to be retrieved)), delete that token. Make sure all items above and below that token have that token removed from their list of connections (e.g., a to be deleted RB is removed as a connection its parent and child Ps, and its predicate and object POs). 
    new_analog = delete_unretrieved_tokens(new_analog)
    # place copied analog into memory. 
    memory.analogs.append(new_analog)
    # all done.
    return memory

# function to make all tokens from the to be copied analog.
def copy_analog_tokens(analog, new_analog, memory):
    # start with Ps. (1) make the P. (2) then make each RB. Connect the RB to the P. (3) For each RB's POs, (4) check if a PO by that name already exists in new_analog.myPOs, and if so, connect that PO to currentRB, otherwise, make the PO, and connect it to the RB. Then, for each RB without Ps, start with (3) above. Then, for each PO without RBs, start with (4) above.
    for myP in analog.myPs:
        # make a copy of the P.
        copy_P = dataTypes.PUnit(myP.name, myP.set, new_analog, False, new_analog)
        # put the copied P in new_analog and in memory.
        new_analog.myPs.append(copy_P)
        memory.Ps.append(copy_P)
        # make copy_P's RB units.
        for myRB in myP.myRBs:
            # make a copy of the RB.
            copy_RB = dataTypes.RBUnit(myRB.name, myRB.set, new_analog, False, new_analog)
            # put the copy_RB in new_analog and in memory.
            new_analog.myRBs.append(copy_RB)
            memory.RBs.append(copy_RB)
            # connect the copy_RB to copy_P and vise versa.
            copy_RB.myParentPs.append(copy_P)
            copy_P.myRBs.append(copy_RB)
            # make the RBs pred (if it does not already exist). Check if a pred with the same name as myRB.myPred[0] already exists in new_analog.myPOs.
            make_new_PO = True
            for myPO in new_analog.myPOs:
                if myPO.name == myRB.myPred[0].name:
                    # the PO is already in new_analog, so just connect it to the copy_RB.
                    myPO.myRBs.append(copy_RB)
                    copy_RB.myPred.append(myPO)
                    # set make_new_PO flag to False.
                    make_new_PO = False
                    break
            # if the PO does not already exist in the new_analog, then make it.
            if make_new_PO:
                # make the RB's pred.
                copy_pred = dataTypes.POUnit(myRB.myPred[0].name, myRB.myPred[0].set, new_analog, False, new_analog, 1)
                # put the copy_pred in new_analog and in memory.
                new_analog.myPOs.append(copy_pred)
                memory.POs.append(copy_pred)
                # connect the copy_pred to copy_RB and vise versa.
                copy_pred.myRBs.append(copy_RB)
                copy_RB.myPred.append(copy_pred)
                # make all the semantic connections for copy_pred.
                for link in myRB.myPred[0].mySemantics:
                    # create a new link for the copy_pred.
                    new_link = dataTypes.Link(copy_pred, None, link.mySemantic, link.weight)
                    # add the new_link to memory.Links, new_pred.semantics, and link.mySemantic.myPOs.
                    memory.Links.append(new_link)
                    copy_pred.mySemantics.append(new_link)
                    link.mySemantic.myPOs.append(new_link)
            # make the RBs object (if it does not already exist).
            make_new_PO = True
            for myPO in new_analog.myPOs:
                if myPO.name == myRB.myObj[0].name:
                    # the PO is already in new_analog, so just connect it to the copy_RB.
                    myPO.myRBs.append(copy_RB)
                    copy_RB.myObj.append(myPO)
                    # set make_new_PO flag to False.
                    make_new_PO = False
                    break
            # if the PO does not already exist in the new_analog, then make it.
            if make_new_PO:
                # make the RB's object.
                copy_obj = dataTypes.POUnit(myRB.myObj[0].name, myRB.myObj[0].set, new_analog, False, new_analog, 0)
                # put the copy_obj in new_analog and in memory.
                new_analog.myPOs.append(copy_obj)
                memory.POs.append(copy_obj)
                # connect the copy_obj to copy_RB and vise versa.
                copy_obj.myRBs.append(copy_RB)
                copy_RB.myObj.append(copy_obj)
                # make all the semantic connections for copy_obj.
                for link in myRB.myObj[0].mySemantics:
                    # create a new link for the copy_obj.
                    new_link = dataTypes.Link(copy_obj, None, link.mySemantic, link.weight)
                    # add the new_link to memory.Links, copy_obj.semantics, and link.mySemantic.myPOs.
                    memory.Links.append(new_link)
                    copy_obj.mySemantics.append(new_link)
                    link.mySemantic.myPOs.append(new_link)
    # now make all RBs that don't have Ps.
    for myRB in analog.myRBs:
        if len(myRB.myParentPs) == 0:
            # make a copy of the RB.
            copy_RB = dataTypes.RBUnit(myRB.name, myRB.set, new_analog, False, new_analog)
            # put the copy_RB in new_analog and in memory.
            new_analog.myRBs.append(copy_RB)
            memory.RBs.append(copy_RB)
            # make the RBs pred (if it does not already exist). Check if a pred with the same name as myRB.myPred[0] already exists in new_analog.myPOs.
            make_new_PO = True
            for myPO in new_analog.myPOs:
                if myPO.name == myRB.myPred[0].name:
                    # the PO is already in new_analog, so just connect it to the copy_RB.
                    myPO.myRBs.append(copy_RB)
                    copy_RB.myPred.append(myPO)
                    # set make_new_PO flag to False.
                    make_new_PO = False
                    break
            # if the PO does not already exist in the new_analog, then make it.
            if make_new_PO:
                # make the RB's pred.
                copy_pred = dataTypes.POUnit(myRB.myPred[0].name, myRB.myPred[0].set, new_analog, False, new_analog, 1)
                # put the copy_pred in new_analog and memory.
                new_analog.myPOs.append(copy_pred)
                memory.POs.append(copy_pred)
                # connect the copy_pred to copy_RB and vise versa.
                copy_pred.myRBs.append(copy_RB)
                copy_RB.myPred.append(copy_pred)
                # make all the semantic connections for copy_pred.
                for link in myRB.myPred[0].mySemantics:
                    # create a new link for the copy_pred.
                    new_link = dataTypes.Link(copy_pred, None, link.mySemantic, link.weight)
                    # add the new_link to memory.Links, new_pred.semantics, and link.mySemantic.myPOs.
                    memory.Links.append(new_link)
                    copy_pred.mySemantics.append(new_link)
                    link.mySemantic.myPOs.append(new_link)
            # make the RBs object (if it does not already exist).
            make_new_PO = True
            for myPO in new_analog.myPOs:
                if myPO.name == myRB.myObj[0].name:
                    # the PO is already in new_analog, so just connect it to the copy_RB.
                    myPO.myRBs.append(copy_RB)
                    copy_RB.myObj.append(myPO)
                    # set make_new_PO flag to False.
                    make_new_PO = False
                    break
            # if the PO does not already exist in the new_analog, then make it.
            if make_new_PO:
                # make the RB's object.
                copy_obj = dataTypes.POUnit(myRB.myObj[0].name, myRB.myObj[0].set, new_analog, False, new_analog, 0)
                # put the copy_obj in new_analog and memory.
                new_analog.myPOs.append(copy_obj)
                memory.POs.append(copy_obj)
                # connect the copy_obj to copy_RB and vise versa.
                copy_obj.myRBs.append(copy_RB)
                copy_RB.myObj.append(copy_obj)
                # make all the semantic connections for copy_obj.
                for link in myRB.myObj[0].mySemantics:
                    # create a new link for the copy_obj.
                    new_link = dataTypes.Link(copy_obj, None, link.mySemantic, link.weight)
                    # add the new_link to memory.Links, copy_obj.semantics, and link.mySemantic.myPOs.
                    memory.Links.append(new_link)
                    copy_obj.mySemantics.append(new_link)
                    link.mySemantic.myPOs.append(new_link)
    # make all POs that don't have RBs.
    for myPO in analog.myPOs:
        if len(myPO.myRBs) == 0:
            make_new_PO = True
            for checkPO in new_analog.myPOs:
                if checkPO.name == myPO.name:
                    # the PO is already in new_analog, so set make_new_PO flag to False.
                    make_new_PO = False
                    break
            # if the PO does not already exist in the new_analog, then make it.
            if make_new_PO:
                # make the RB's object.
                copy_obj = dataTypes.POUnit(myPO.name, myPO.set, new_analog, False, new_analog, 0)
                # put the copy_obj in new_analog and memory.
                new_analog.myPOs.append(copy_obj)
                memory.POs.append(copy_obj)
                # make all the semantic connections for copy_obj.
                for link in myPO.mySemantics:
                    # create a new link for the copy_obj.
                    new_link = dataTypes.Link(copy_obj, None, link.mySemantic, link.weight)
                    # add the new_link to memory.Links, copy_obj.semantics, and link.mySemantic.myPOs.
                    memory.Links.append(new_link)
                    copy_obj.mySemantics.append(new_link)
                    link.mySemantic.myPOs.append(new_link)
    # all done.
    return new_analog, memory

# function to clear .set field for all tokens in an analog.
def clear_set(analog):
    for Group in analog.myGroups:
        Group.set = 'memory'
    for myP in analog.myPs:
        myP.set = 'memory'
    for myRB in analog.myRBs:
        myRB.set = 'memory'
    for myPO in analog.myPOs:
        myPO.set = 'memory'
    # done.
    return analog

# update newSet inputs.
def update_newSet_inputs(memory):
    # units in NewSet have input 1 if the token that made them in the driver is active above threshold, 0 otherwise.
    threshold = .75
    for Group in memory.newSet.Groups:
        if Group.my_maker_unit.act > threshold:
            Group.act = 1.0
        else:
            Group.act = 0.0
    for myP in memory.newSet.Ps:
        if myP.my_maker_unit.act > threshold:
            myP.act = 1.0
        else:
            myP.act = 0.0
    for myRB in memory.newSet.RBs:
        if myRB.my_maker_unit:
            if myRB.my_maker_unit.act > threshold:
                myRB.act = 1.0
            else:
                myRB.act = 0.0
    for myPO in memory.newSet.POs:
        if myPO.my_maker_unit:
            if myPO.my_maker_unit.act > threshold:
                myPO.act = 1.0
            else:
                myPO.act = 0.0
    # done.
    return memory

# function to make sure all lower tokens of a to be retrieved into AM token in an analog are also set to be retrieved.
def retrieve_all_relevant_tokens(analog):
    # check each token, and if it is to be retrieved into AM (i.e., .set is NOT 'memory), make sure all tokens below it are also be be retrieved into AM.
    for Group in analog.myGroups:
        if Group.set != 'memory':
            Group = retrieve_lower_tokens(Group)
    for myP in analog.myPs:
        if myP.set != 'memory':
            myP = retrieve_lower_tokens(myP)
    for myRB in analog.myRBs:
        if myRB.set != 'memory':
            myRB = retrieve_lower_tokens(myRB)
    for myPO in analog.myPOs:
        if myPO.set != 'memory':
            myPO = retrieve_lower_tokens(myPO)
    # done.
    return analog

# function to make sure all a token's sub-tokens are in the proper .set
def retrieve_lower_tokens(token):
    # check what kind of token you're dealing with.
    if token.my_type == 'Group':
        # if you're dealing with a Group, then for each of it's sub-groups, sub-Ps, and sub-RBs, set that sub_token.set to the same set as the Group, and run retrieve_lower_tokens on that sub-token.
        for Group_under in token.myChildGroups:
            Group_under.set = token.set
            Group_under = retrieve_lower_tokens(Group_under)
        for myP in token.myPs:
            myP.set = token.set
            myP = retrieve_lower_tokens(myP)
        for myRB in token.myRBs:
            myRB.set = token.set
            myRB = retrieve_lower_tokens(myRB)
    if token.my_type == 'P':
        # if you're dealing with a P, then for each of it's RBs, set that myRB.set to the same set as the P, and run retrieve_lower_tokens on the myRB.
        for myRB in token.myRBs:
            myRB.set = token.set
            myRB = retrieve_lower_tokens(myRB)
    if token.my_type == 'RB':
        # if you're dealing with a RB, then for each of it's child-Ps and POs, set that token.set to the same set as the RB, and run retrieve_lower_tokens on the the sub-token.
        token.myPred[0].set = token.set
        token.myPred[0] = retrieve_lower_tokens(token.myPred[0])
        if len(token.myObj) > 0:
            token.myObj[0].set = token.set
            token.myObj[0] = retrieve_lower_tokens(token.myObj[0])
        elif len(token.myChildP) > 0:
            token.myChildP[0].set = token.set
            token.myChildP[0] = retrieve_lower_tokens(token.myChildP[0])
    # done.
    return token

# function to delete unretrieved tokens from a copied analog.
def delete_unretrieved_tokens(analog):
    # go through each token in the analog. If it is unretrieved (i.e., token.set == 'memory'), delete that token and make sure you also delete that token from any tokens to which is is connected. NOTE: You don't need to worry about connections between POs and semantics, as the semantics copied POs are connected to are themselves copied and it doesn't matter if they are deleted. You'll replace these copied semantics with the original semantics using replace_copied_semantics() later in the check_analog_for_tokens_to_copy() function. 
    for Group in analog.myGroups:
        if Group.set == 'memory':
            analog = delete_token(Group, analog)
    for myP in analog.myPs:
        if myP.set == 'memory':
            analog = delete_token(myP, analog)
    for myRB in analog.myRBs:
        if myRB.set == 'memory':
            analog = delete_token(RB, analog)
    for myPO in analog.myPOs:
        if myPO.set == 'memory':
            analog = delete_token(myPO, analog)
    # done.
    return analog

# function to delete a token from an analog.
def delete_token(token, analog):
    # figure out what kind of unit token is, then delete the token and delete instances of that token from any units it is connected to.
    if token.my_type == 'Group':
        # delete the Group from its ParentGroups, ChildGroups, Ps, and RBs.
        for pGroup in token.myParentGroups:
            pGroup.myChildGroups.remove(token)
        for cGroup in token.myChildGroups:
            cGroup.myParentGroups.remove(token)
        for myP in token.myPs:
            myP.myGroups.remove(token)
        for myRB in token.myRBs:
            myRB.myGroups.remove(token)
        # delete the Group iteself from analog.
        # NOTE: You don't need to delete the analog from memory, because you haven't added the analog to memory yet. It is still just a copied analog
        analog.myGroups.remove(token)
    elif token.my_type == 'P':
        # delete the P from its Groups, ParentRBs, and ChildRBs.
        for Group in token.myGroups:
            Group.myPs.remove(token)
        for ParentRB in token.myParentRBs:
            ParentRB.myChildP.remove(token)
        for ChildRB in token.myRBs:
            ChildRB.myParentPs.remove(token)
        # delete the P iteself.	
        analog.myPs.remove(token)
    elif token.my_type == 'RB':
        # delete the RB from its ParentPs, Pred, and either ChildP or Object.
        for ParentP in token.myParentPs:
            ParentP.myRBs.remove(token)
        for pred in token.myPred:
            # if you are removing an RB, make sure that the PO connected to that RB also has the RB's second PO removed from its .same_RB_POs field. That is, a PO knows what other POs share RBs with it (so that it does not inhibit them during LISA mode). If an RB is deleted, the two POs sharing that RB no longer do. (RECALL: in a single analog, if two instances of a role-binding occur, the same PO tokens and same RB token is used to instantiate both (e.g., if L1(x) occurs twice in the same analog, then L1, x and L1+x are used in both instances). As a consequence, if an RB is deleted, then it means that the Pred and object are not linked by an RB in the current analog, and, as such, that neither PO should appear in the other PO's .same_RB_POs field.)
            # first, check to make sure the RB takes an object as an argument. A P as an argument will not appear in myPO.same_RB_POs as a P is NOT a myPO.
            if len(token.myObj) > 0:
                pred.same_RB_POs.remove(token.myObj[0])
            pred.myRBs.remove(token)
        for ChildP in token.myChildP:
            ChildP.myParentRBs.remove(token)
        for obj in token.myObj:
            # delete the token's pred from the object's .same_RB_POs field.
            obj.same_RB_POs.remove(token.myPred[0])
            obj.myRBs.remove(token)
        # delete the RB iteself.
        analog.myRBs.remove(token)
    elif token.my_type == 'PO':
        # delete the PO from its RBs.
        for myRB in token.myRBs:
            if token.predOrObj == 1:
                myRB.myPred.remove(token)
            else:
                myRB.myObj.remove(token)
        # delete the PO iteself.
        analog.myPOs.remove(token)
    # done.
    return analog

# function to replace semantics in a new PO with the original memory.semantics versions.
def replace_copied_semantics(myPO, semantics):
    # for each Link in my .mySemantics, find the original semantic with the same name as .mySemantic.name, and replace .mySemantic with the original semantic from memory.
    for Link in myPO.mySemantics:
        # search each semantic in memory.semantics for the one with the same name as Link.mySemantic. Once you have found that semantic, replace Link.mySemantic with the semantic from memory, and break the for loop. 
        for semantic in semantics:
            if semantic.name == Link.mySemantic.name:
                Link.mySemantic = semantic
                semantic.myPOs.append(Link)
                break # break the for loop.
    # done. 
    return PO, semantics

# function to find token in memory whose set is driver or recipient in order to construct the driver and recipient sets for the run. Returns driver and recipient sets. 
def findDriverRecipient(memory):
    # first clear out the memory.driver and memory.recipient fields.
    memory.driver.Groups, memory.driver.Ps, memory.driver.RBs, memory.driver.POs, memory.driver.analogs = [], [], [], [], []
    memory.recipient.Groups, memory.recipient.Ps, memory.recipient.RBs, memory.recipient.POs, memory.recipient.analogs = [], [], [], [], []
    # for each Group, P, RB, and PO, if the set is driver, put in in the driverSet, otherwise, elif the set is recipient, put it in the recipientSet.
    for Group in memory.Groups:
        if Group.set == 'driver':
            memory.driver.Groups.append(Group)
            # reset the .copy_for_DR field back to False.
            Group.copy_for_DR = False
            # now add the analog to driver.analogs if it is not already there.
            if Group.myanalog not in memory.driver.analogs:
                memory.driver.analogs.append(Group.myanalog)
        elif Group.set == 'recipient':
            memory.recipient.Groups.append(Group)
            # reset the .copy_for_DR field back to False.
            Group.copy_for_DR = False
            # now add the analog to recipient.analogs if it is not already there.
            if Group.myanalog not in memory.recipient.analogs:
                memory.recipient.analogs.append(Group.myanalog)
    for myP in memory.Ps:
        if myP.set == 'driver':
            memory.driver.Ps.append(myP)
            # reset the .copy_for_DR field back to False.
            myP.copy_for_DR = False
            # now add the analog to driver.analogs if it is not already there.
            if myP.myanalog not in memory.driver.analogs:
                memory.driver.analogs.append(myP.myanalog)
        elif myP.set == 'recipient':
            memory.recipient.Ps.append(myP)
            # reset the .copy_for_DR field back to False.
            myP.copy_for_DR = False
            # now add the analog to recipient.analogs if it is not already there.
            if myP.myanalog not in memory.recipient.analogs:
                memory.recipient.analogs.append(myP.myanalog)
    for myRB in memory.RBs:
        if myRB.set == 'driver':
            memory.driver.RBs.append(myRB)
            # reset the .copy_for_DR field back to False.
            myRB.copy_for_DR = False
            # now add the analog to driver.analogs if it is not already there.
            if myRB.myanalog not in memory.driver.analogs:
                memory.driver.analogs.append(myRB.myanalog)
        elif myRB.set == 'recipient':
            memory.recipient.RBs.append(myRB)
            # reset the .copy_for_DR field back to False.
            myRB.copy_for_DR = False
            # now add the analog to recipient.analogs if it is not already there.
            if myRB.myanalog not in memory.recipient.analogs:
                memory.recipient.analogs.append(myRB.myanalog)
    for myPO in memory.POs:
        if myPO.set == 'driver':
            memory.driver.POs.append(myPO)
            # now add the analog to driver.analogs if it is not already there.
            if myPO.myanalog not in memory.driver.analogs:
                memory.driver.analogs.append(myPO.myanalog)
        elif myPO.set == 'recipient':
            memory.recipient.POs.append(myPO)
            # now add the analog to recipient.analogs if it is not already there.
            if myPO.myanalog not in memory.recipient.analogs:
                memory.recipient.analogs.append(myPO.myanalog)
    # done.
    return memory

# make firing order.
def makeFiringOrder(memory, rule):
    # set the firing order of the driver using rule.
    # right now, the only rule is random, the default.
    # you should add pragmatics.
    if rule == 'by_top_random':
        # arrange RBs randomly within Ps or Groups.
        if len(memory.driver.Groups) > 0:
            # arrange by Groups.
            # randomly arrange the Groups.
            Gorder = memory.driver.Groups
            random.shuffle(Gorder)
            # now select RBs from Porder.
            firingOrder = []
            Porder = []
            for Group in Gorder:
                # order my Ps.
                for myP in Group.myPs:
                    Porder.append(myP)
            # now add the RBs from each P in Porder to firingOrder.
            for myP in Porder:
                for myRB in myP.myRBs:
                    # add RB to firingOrder.
                    firingOrder.append(myRB)
        elif len(memory.driver.Ps) > 0: # arrange by Ps.
            # randomly arrange the Ps.
            Porder = memory.driver.Ps
            random.shuffle(myPorder)
            # now select RBs from Porder.
            firingOrder = []
            for myP in Porder:
                for myRB in myP.myRBs:
                    # add RB to firingOrder.
                    firingOrder.append(myRB)
        else:
            # arrange RBs or POs randomly.
            firingOrder = []
            if len(memory.driver.RBs) > 0:
                for myRB in memory.driver.RBs:
                    firingOrder.append(myRB)
                random.shuffle(firingOrder)
            else:
                # arrange by POs.
                for myPO in memory.driver.POs:
                    firingOrder.append(myPO)
                random.shuffle(firingOrder)
    else: # use a totally random firing order.
        if not rule == 'totally_random':
            print 'You have not input a valid firing rule. I am arranging RBs at random.'
        firingOrder = []
        if len(memory.driver.RBs) > 0:
            for myRB in memory.driver.RBs:
                firingOrder.append(myRB)
            random.shuffle(firingOrder)
        else:
            # arrange by POs.
            for myPO in memory.driver.POs:
                firingOrder.append(myPO)
            random.shuffle(firingOrder)
    # done.
    return firingOrder

# index all items in memory.
def indexMemory(memory):
    for Group in memory.Groups:
        Group.get_index(memory)
    for myP in memory.Ps:
        myP.get_index(memory)
    for myRB in memory.RBs:
        myRB.get_index(memory)
    for myPO in memory.POs:
        myPO.get_index(memory)
    # done.
    return memory

# update all the .same_RB_POs for all POs in memory.
def update_same_RB_POs(memory):
    # clear the same_RB_PO field of all POs in memory.
    for myPO in memory.POs:
        myPO.same_RB_POs = []
    # update the .same_RB_PO field of POs by iterating through RBs, and adding objects to pred's field and preds to object's field.
    for myRB in memory.RBs:
        # if there is an object and a pred, add the pred to object's .same_RB_POs, and object to pred's .same_RB_POs.
        if (len(myRB.myObj) > 0) and (len(myRB.myPred) > 0):
            myRB.myObj[0].same_RB_POs.append(myRB.myPred[0])
            myRB.myPred[0].same_RB_POs.append(myRB.myObj[0])
    # done.
    return memory

# a function to clear activation and input to all driver, recipient, newSet, and semantic units (i.e., everything in active memory, AM).
def initialize_AM(memory):
    for Group in memory.driver.Groups:
        Group.initialize_act()
    for myP in memory.driver.Ps:
        myP.initialize_act()
    for myRB in memory.driver.RBs:
        myRB.initialize_act()
    for myPO in memory.driver.POs:
        myPO.initialize_act()
    for Group in memory.recipient.Groups:
        Group.initialize_act()
    for myP in memory.recipient.Ps:
        myP.initialize_act()
    for myRB in memory.recipient.RBs:
        myRB.initialize_act()
    for myPO in memory.recipient.POs:
        myPO.initialize_act()
    for Group in memory.newSet.Groups:
        Group.initialize_act()
    for myP in memory.newSet.Ps:
        myP.initialize_act()
    for myRB in memory.newSet.RBs:
        myRB.initialize_act()
    for myPO in memory.newSet.POs:
        myPO.initialize_act()
    for semantic in memory.semantics:
        semantic.initializeSem()
    # done.
    return memory

# a function to clear the activation and input to all units in the network.
def initialize_memorySet(memory):
    for Group in memory.Groups:
        Group.initialize_act()
    for myP in memory.Ps:
        myP.initialize_act()
    for myRB in memory.RBs:
        myRB.initialize_act()
    for myPO in memory.POs:
        myPO.initialize_act()
    # done.
    return memory

# initialize input to all driver, recipient, newSet and semantic units.
def initialize_input(memory):
    for Group in memory.driver.Groups:
        Group.initialize_input(0.0)
    for myP in memory.driver.Ps:
        myP.initialize_input(0.0)
    for myRB in memory.driver.RBs:
        myRB.initialize_input(0.0)
    for myPO in memory.driver.POs:
        myPO.initialize_input(0.0)
    for Group in memory.recipient.Groups:
        Group.initialize_input(0.0)
    for myP in memory.recipient.Ps:
        myP.initialize_input(0.0)
    for myRB in memory.recipient.RBs:
        myRB.initialize_input(0.0)
    for myPO in memory.recipient.POs:
        myPO.initialize_input(0.0)
    for Group in memory.newSet.Groups:
        Group.initialize_input(0.0)
    for myP in memory.newSet.Ps:
        myP.initialize_input(0.0)
    for myRB in memory.newSet.RBs:
        myRB.initialize_input(0.0)
    for myPO in memory.newSet.POs:
        myPO.initialize_input(0.0)
    for semantic in memory.semantics:
        semantic.initialize_input(0.0)
    # done.
    return memory

# update the activations of all units in driver, recipient, and newSet.
def update_activations_run(memory, gamma, delta, HebbBias, phase_set, do_ding=False):
    for Group in memory.driver.Groups:
        Group.update_act(gamma, delta, HebbBias)
    for myP in memory.driver.Ps:
        myP.update_act(gamma, delta, HebbBias)
    for myRB in memory.driver.RBs:
        myRB.update_act(gamma, delta, HebbBias)
    for myPO in memory.driver.POs:
        myPO.update_act(gamma, delta, HebbBias)
    for Group in memory.recipient.Groups:
        Group.update_act(gamma, delta, HebbBias)
    for myP in memory.recipient.Ps:
        myP.update_act(gamma, delta, HebbBias)
    for myRB in memory.recipient.RBs:
        myRB.update_act(gamma, delta, HebbBias)
    for myPO in memory.recipient.POs:
        myPO.update_act(gamma, delta, HebbBias)
    for Group in memory.newSet.Groups:
        Group.update_act(gamma, delta, HebbBias)
    for myP in memory.newSet.Ps:
        myP.update_act(gamma, delta, HebbBias)
    for myRB in memory.newSet.RBs:
        myRB.update_act(gamma, delta, HebbBias)
    for myPO in memory.newSet.POs:
        myPO.update_act(gamma, delta, HebbBias)
    # get the max input to any semantic unit, then update semantic activations.
    if not do_ding:
        max_input = get_max_sem_input(memory)
        for semantic in memory.semantics:
            semantic.set_max_input(max_input)
            semantic.update_act()
    # done.
    return memory

# update the activation of all units in memory that are NOT in driver, recipient, or newSet. (For use in retrieval.)
def update_acts_memory(memory, gamma, delta, HebbBias):
    for Group in memory.Groups:
        if Group.set == 'memory':
            Group.update_act(gamma, delta, HebbBias)
    for myP in memory.Ps:
        if myP.set == 'memory':
            myP.update_act(gamma, delta, HebbBias)
    for myRB in memory.RBs:
        if myRB.set == 'memory':
            myRB.update_act(gamma, delta, HebbBias)
    for myPO in memory.POs:
        if myPO.set == 'memory':
            myPO.update_act(gamma, delta, HebbBias)
    # done.
    return memory

# update inputs to driver units.
def update_driver_inputs(memory, asDORA, lateral_input_level):
    # update inputs to all driver units.
    for Group in memory.driver.Groups:
        Group.update_input_driver(memory, asDORA)
    for myP in memory.driver.Ps:
        if myP.mode == 1:
            myP.update_input_driver_parent(memory, asDORA)
        elif myP.mode == -1:
            myP.update_input_driver_child(memory, asDORA)
    for myRB in memory.driver.RBs:
        myRB.update_input_driver(memory, asDORA)
    for myPO in memory.driver.POs:
        myPO.update_input_driver(memory, asDORA)
    # done
    return memory

# update inputs to recipient units.
def update_recipient_inputs(memory, asDORA, phase_set, lateral_input_level, ignore_object_semantics):
    # update inputs to all recipient units.
    for Group in memory.recipient.Groups:
        Group.update_input_driver(memory, asDORA)
    for myP in memory.recipient.Ps:
        if myP.mode == 1:
            myP.update_input_recipient_parent(memory, asDORA, phase_set, lateral_input_level)
        elif myP.mode == -1:
            myP.update_input_recipient_child(memory, asDORA, phase_set, lateral_input_level)
    for myRB in memory.recipient.RBs:
        myRB.update_input_recipient(memory, asDORA, phase_set, lateral_input_level)
    for myPO in memory.recipient.POs:
        myPO.update_input_recipient(memory, asDORA, phase_set, lateral_input_level, ignore_object_semantics)
    # done.
    return memory

# update input to all memorySet units that are not in driver, recipient, or newSet (used during retreival).
def update_memory_inputs(memory, asDORA, lateral_input_level):
    # for all units not in driver, recipient, or newSet (i.e., units with set != driver, recipient, or newSet), update input. Units in memory update as units in recipient.
    # set phase_set to 2.
    phase_set = 2
    for Group in memory.Groups:
        if Group.set == 'memory':
            Group.update_input_recipient(memory, asDORA, phase_set, lateral_input_level)
    for myP in memory.Ps:
        if myP.set == 'memory':
            # NOTE: I think it might be best to avoid modes altogether when working in retieval mode. This version of the code reflects this assumption.
            myP.update_input_recipient_parent(memory, asDORA, phase_set, lateral_input_level)
    for myRB in memory.RBs:
        if myRB.set == 'memory':
            myRB.update_input_recipient(memory, asDORA, phase_set, lateral_input_level)
    for myPO in memory.POs:
        if myPO.set == 'memory':
            myPO.update_input_recipient(memory, asDORA, phase_set, lateral_input_level) # update with phase_set = 2 so that myPO units also take top down input from RBs.
    # done.
    return memory

# get the max input to semantics unit in the network.
def get_max_sem_input(memory):
    max_input = 0.0
    for semantic in memory.semantics:
        if semantic.myinput > max_input:
            max_input = semantic.myinput
    # done.
    return max_input

# function to do run the network during retieval.
def retrieval_routine(memory, asDORA, gamma, delta, HebbBias, lateral_input_level, bias_retrieval_analogs):
    # update input to memorySet units.
    memory = update_memory_inputs(memory, asDORA, lateral_input_level)
    # update activation of memorySet units.
    memory = update_acts_memory(memory, gamma, delta, HebbBias)
    if bias_retrieval_analogs:
        # for each analog, track the total activation of its units if they are in memory (i.e., if the analog is not already in driver or recipient). 
        for analog in memory.analogs:
            analog.total_act = 0.0
            for myP in analog.myPs:
                if myP.set == 'memory':
                    analog.total_act += myP.act
            for myRB in analog.myRBs:
                if myRB.set == 'memory':
                    analog.total_act += myRB.act
            for myPO in analog.myPOs:
                if myPO.set == 'memory':
                    analog.total_act += myPO.act
            analog.sum_num_units()
    else:
        # track the most active P, RB, and PO units in memory.
        for myP in memory.Ps:
            if myP.set == 'memory':
                if myP.act > myP.max_act:
                    myP.max_act = myP.act
        for myRB in memory.RBs:
            if myRB.set == 'memory':
                if myRB.act > myRB.max_act:
                    myRB.max_act = myRB.act
        for myPO in memory.POs:
            if myPO.set == 'memory':
                if myPO.act > myPO.max_act:
                    myPO.max_act = myPO.act
        # done.
    return memory

# function to retrieve tokens from memory. Takes as arguments the memory set, and a bias_retrieval_analogs flag that if True, biases retrieval towards whole analogs.
def retrieve_tokens(memory, bias_retrieval_analogs, use_relative_act):    
    # if bias_retrieval_analogs is true, bias towards retrieving whole analogs. Otherwise, default to no bias (myPs, RBs, and POs stand some odds of being retrieved regardless of their interconnectivity (of course, if a token is retrieved, all tokens below it that the token is connected to are also retrieved)). 
    if use_relative_act:
        # retrieve using relative activation of propositions.
        if bias_retrieval_analogs:
            # retrieve whole analogs.
            # create a normalised retrieval score for each analog (i.e., analog.total_act/analog.num_units), and make a list of all analog activations.
            analog_activation_list = []
            for analog in memory.analogs:
                # make sure analog has a .total_act and .num_units > 0.
                if analog.total_act > 0 and analog.num_units > 0:
                    # calculate analog.normalised_retrieval_act and add that to sum_normalised_analogs.
                    analog.normalised_retrieval_act = analog.total_act/analog.num_units
                    analog_activation_list.append(analog.normalised_retrieval_act)
            # retrieve analogs with a probability calculated as a function of the ratio of the specific analog's normalised activation to the average normalised activation of all active analogs.
            # find the average and highest normalised activation for analogs.
            avg_analog_norm_act = np.mean(analog_activation_list)
            high_analog_norm_act = max(analog_activation_list)
            avg_analog_norm_act = (high_analog_norm_act+avg_analog_norm_act)/2
            #pdb.set_trace()
            # transform all retrieval activations using a sigmoidal function with a threshold around high_analog_norm_act. 
            for analog in memory.analogs:
                if analog.total_act > 0:
                    analog.normalised_retrieval_act = 1/(1 + math.exp(10*(analog.normalised_retrieval_act-avg_analog_norm_act)))
            # get the sum of all transformed noralised analog activations. 
            sum_analog_norm_act = sum(analog_activation_list)
            # retrieve analogs using the Luce choice rule appled to transformed activations. 
            for analog in memory.analogs:
                # if analog has a .total_act and .num_units > 0, then calculate the retrieve_prob.
                if analog.total_act > 0 and analog.num_units > 0:
                    retrieve_prob = analog.normalised_retrieval_act/sum_analog_norm_act
                    randomNum = random.random()
                    if retrieve_prob >= randomNum:
                        # retrieve the analog and all it's tokens.
                        analog = retrieve_analog_contents(analog)
    else:
        # retirieve using the old Luce choice axiom. 
        if bias_retrieval_analogs:
            # retrieve whole analogs.
            # create a normalised retrieval score for each analog (i.e., analog.total_act/analog.num_units) and sum up all normalised retrieval scores for each analog in memory.
            sum_normalised_analogs = 0.0
            for analog in memory.analogs:
                # make sure analog has a .total_act and .num_units > 0.
                if analog.total_act > 0 and analog.num_units > 0:
                    # calculate my num_units.
                    analog.sum_num_units()
                    # calculate analog.normalised_retrieval_act and add that to sum_normalised_analogs.
                    analog.normalised_retrieval_act = analog.total_act/analog.num_units
                    sum_normalised_analogs += analog.normalised_retrieval_act
            # retrieve analogs using the Luce choice axiom.
            for analog in memory.analogs:
                # if analog has a .total_act and .num_units > 0, then calculate the retrieve_prob via Luce choice.
                if analog.total_act > 0 and analog.num_units > 0:
                    retrieve_prob = analog.normalised_retrieval_act/sum_normalised_analogs
                    randomNum = random.random()
                    if retrieve_prob >= randomNum:
                        # retrieve the analog and all it's tokens.
                        analog = retrieve_analog_contents(analog)
        else:
            # get sum of all max_acts of all P, RB and P units in memorySet. 
            P_sum, RB_sum, PO_sum = 0.0, 0.0, 0.0
            for myP in memory.Ps:
                P_sum += myP.max_act
            for myRB in memory.RBs:
                RB_sum += myRB.max_act
            for myPO in memory.POs:
                PO_sum += myPO.max_act
            # for each P, RB, and PO in memorySet (i.e., NOT in driver, recipient, or newSet), retrieve it (and the proposition attached to it) into recipient according to the Luce choice rule. 
            # P units.
            for myP in memory.Ps:
                # make sure that the P is in memory and that P_sum > 0 (so you don't get a divide by 0 error). 
                if (myP.set == 'memory') and (myP_sum > 0):
                    retrieve_prob = myP.max_act/P_sum
                    randomNum = random.random()
                    if retrieve_prob > randomNum:
                        # retrieve P and all units attached into recipient.
                        myP.set = 'recipient'
                        # add the RBs.
                        for myRB in myP.myRBs:
                            myRB.set = 'recipient'
                            # add the POs.
                            myRB.myPred[0].set = 'recipient'
                            # if it has an object add that object.
                            if len(myRB.myObj) >= 1:
                                myRB.myObj[0].set = 'recipient'
                            else: # add it's child myP.
                                myRB.myChildP[0].set = 'recipient'
            # RB units.
            for myRB in memory.RBs:
                # make sure that the RB is in memory and that RB_sum > 0 (so you don't get a divide by 0 error). 
                if (RB.set == 'memory') and (RB_sum > 0):
                    retrieve_prob = myRB.max_act/RB_sum
                    randomNum = random.random()
                    if retrieve_prob > randomNum:
                        # retrieve RB and all units attached into recipient.
                        myRB.set = 'recipient'
                        # add the Ps.
                        for myP in myRB.myParentPs:
                            myP.set = 'recipient'
                        # add the POs.
                        myRB.myPred[0].set = 'recipient'
                        # if it has an object add that object.
                        if len(RB.myObj) >= 1:
                            myRB.myObj[0].set = 'recipient'
                        else: # add it's child myP.
                            myRB.myChildP[0].set = 'recipient'
            # PO units.
            for myPO in memory.POs:
                # make sure that the PO is in memory and that PO_sum > 0 (so you don't get a divide by 0 error). 
                if (myPO.set == 'memory') and (myPO_sum > 0):
                    retrieve_prob = myPO.max_act/PO_sum
                    randomNum = random.random()
                    if retrieve_prob > randomNum:
                        # retrieve PO and all units attached into recipient.
                        myPO.set = 'recipient'
                        memory.recipient.POs.append(myPO)
                        # add the RBs.
                        for myRB in myPO.myRBs:
                            myRB.set = 'recipient'
                            # add the RB's P unit if it exists.
                            if len(myRB.myParentPs) > 0:
                                myRB.myParentPs[0].set = 'recipient'
    # done.
    return memory

# funtion to retrieve all of the contents of an analog from memory into the recipient. 
def retrieve_analog_contents(analog):
    for myP in analog.myPs:
        myP.set = 'recipient'
    for myRB in analog.myRBs:
            myRB.set = 'recipient'
    for myPO in analog.myPOs:
        myPO.set = 'recipient'

# Take as input a set of nodes of a specific type (e.g., memory.POs, or memory.recipient.RBs) and return most active unit.
def get_most_active_unit(tokens):
    # make sure that you've passed a non-empty array.
    if len(tokens) > 0:
        active_token = tokens[0]
        for token in tokens:
            if token.act > active_token.act:
                active_token = token
        # make sure that you're actually returning an active unit (not just the first token if all token have the same activation (e.g., 0.0)).
        if active_token.act < .01:
            active_token = None
    else:
        active_token = None
    # done.
    return active_token

# Take as input a set of P units and a tag specifying 'parent' or 'child', and return most active unit of that type.
def get_most_active_Punit(tokens, tag):
    if tag == 'parent':
        desired_mode = 1
    elif tag == 'child':
        desired_mode = -1
    else:
        desired_mode = 0
    activity = 0.0
    active_token = None
    for token in tokens:
        if token.act > activity and token.mode == desired_mode:
            active_token = token
            activity = token.act
    # done.
    return active_token

# function to find the analog in the recipient that contains all the mapped recipient units. Currently for use only with rel_gen_routine() function.
def find_recip_analog(memory):
    # search through the POs in the recipient and find their analog. (You only need to search the POs because all recipient units that map have already been compiled into a single analog, and all analogs contain at least POs.)
    for myPO in memory.recipient.POs:
        if myPO.max_map > 0.0:
            recip_analog = myPO.myanalog
            break
    # done.
    return recip_analog

# function to find the analog in the driver that contains all the mapped driver units. Currently for use only with do_rel_gen() routine from the runDORA object.
def find_driver_analog_rel_gen(memory):
    # search through the POs in the driver and find their analog. (You only need to search the POs because all driver units that map are from a single analog, and all analogs contain at least POs.)
    for myPO in memory.driver.POs:
        if myPO.max_map > 0.0:
            driver_analog = myPO.myanalog
            break
    # done.
    return driver_analog

# function to clear the set field of every token in memory (i.e., to clear WM).
def clearTokenSet(memory):
    # for each P, RB, and PO, clear the set field.
    for Group in memory.Groups:
        Group.set = 'memory'
    for myP in memory.Ps:
        myP.set = 'memory'
    for myRB in memory.RBs:
        myRB.set = 'memory'
    for myPO in memory.POs:
        myPO.set = 'memory'
    # done.
    return memory

# function to clear the driver.
def clearDriverSet(memory):
    # for each P, RB, and PO, clear the set field.
    for Group in memory.driver.Groups:
        Group.set = 'memory'
    for myP in memory.driver.Ps:
        myP.set = 'memory'
    for myRB in memory.driver.RBs:
        myRB.set = 'memory'
    for myPO in memory.driver.POs:
        myPO.set = 'memory'
    # now clear the memory.driver fields.
    memory.driver.Ps = []
    memory.driver.RBs = []
    memory.driver.POs = []
    # done.
    return memory

# function to clear the recipient.
def clearRecipientSet(memory):
    # for each P, RB, and PO, clear the set field.
    for Group in memory.recipient.Groups:
        Group.set = 'memory'
    for myP in memory.recipient.Ps:
        myP.set = 'memory'
    for myRB in memory.recipient.RBs:
        myRB.set = 'memory'
    for myPO in memory.recipient.POs:
        myPO.set = 'memory'
    # now clear the memory.recipient fields.
    memory.recipient.Ps = []
    memory.recipient.RBs = []
    memory.recipient.POs = []
    # done.
    return memory

