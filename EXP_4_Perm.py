import numpy as np
import time
import random
from itertools import product 
from psychopy import visual, core, data, event, sound
from psychopy_legacy.visual.ratingscale import RatingScale
import gc
import os

import stimuli

LOG_FLAG = True


def main():
	# Sigma Settings
	SHAPE = ["Circle", "Triangle"]
	FILL = ["Red", "Blue"]
	SIZE = ["Large", "Small"]
	R = 2
	GENERATION_MODE = "Multiset Permutation"

	# Formula Pool
	T1 = [
		   [
			  ("=1", "+0"),
			  ("=1", "+0"),
			  ("=1", "+0"),
		   ]
		 ] 
	T2 = [
		   [
			  ("=1", "+0"),
			  ("+0", "+0"),
			  ("+0", "+0"),
		   ]
		   ,
		   [
			  ("+0", "+0"),
			  ("=1", "+0"),
			  ("=1", "+0"),
		   ]	   
		 ]
	B1 = [
		   [
			  ("=1", "+0"),
			  ("=1", "+0"),
			  ("+0", "+0"),
		   ]
		 ]
	B2 = [
		   [
			  ("=1", "+0"),
			  ("+0", "+0"),
			  ("+0", "+0"),
		   ]
		   ,
		   [
			  ("+0", "+0"),
			  ("=1", "+0"),
			  ("+0", "+0"),
		   ]	   
		 ]
	S1 = [
		   [
			  ("=1", "+0"),
			  ("+0", "+0"),
			  ("+0", "+0"),
		   ]
		 ]
	D2 = [
		   [
			  ("=1", "+0"),
			  ("=1", "+0"),
			  ("+0", "+0"),
		   ]
		 ]
	D3 = [
		   [
			  ("=1", "+0"),
			  ("+0", "+0"),
			  ("=1", "+0"),
		   ]
		 ]
	D4 = [
		   [
			  ("+0", "+0"),
			  ("=1", "+0"),
			  ("=1", "+0"),
		   ]
		 ]
	TF = [
		   [
			  ("=1", "+0"),
			  ("+0", "+0"),
		   ]
		   ,
		   [
			  ("+0", "+0"),
			  ("+0", "=1"),
		   ]	   
		 ]
	GL = [
			(19, 19, 19),
			(19, 13, 19),
			(17, 19, 19),
			(19, 13, 17),
			(2, 2, 2,),
			(3, 2, 2,),
			(2, 19, 2),
			(19, 2, 19),
			(2, 17, 19),
			(17, 17, 2),
			(7, 11, 19),
			(19, 5, 11),
			(11, 13, 7),
			(7, 13, 17),
			(13, 17, 7),
		  ]

	# Output Settings
	DIRECTORY = "./rsps/"
	HEADER = ["Blc", "Seq", "Rsp", "Truth", "Rsp_t1", "Rsp_t2", "Cnfdnc"]

	###########################################################################

	# # Testing Formula
	# test_SIG = stimuli.Sigma([["r", "b"], ["c","t"], ["l","s"]], ["fill", "shape", "size"], R, generation_mode = "Multiset Permutation")
	# # for seq in test_SIG.sequences: 
	# # 	for obj in seq: print(obj)
	# # 	print("########")
	# print(len(test_SIG.sequences))
	# test_conj = test_SIG.form_conjunct(B2, conjunct_type="Seq")
	# acc_count = 0
	# for seq in test_SIG.sequences:
	# 	if test_conj.accepts(seq): acc_count += 1
	# 	print(test_conj.accepts(seq) , seq.summarize())
	# print(acc_count)
	# exit()

	# Init Sigma
	SIG = stimuli.Sigma([FILL, SHAPE, SIZE], ["fill", "shape", "size"], R, generation_mode = GENERATION_MODE)

	# Get input settings
	sub_code = input("Enter Subject code: ")
	while os.path.exists(DIRECTORY + sub_code):
		sub_code = input("Duplicate Subject code. Enter a new code again: ")
	conj_mode = input("Enter Formula Mode: ")
	while conj_mode not in ("T1", "T2", "B1", "B2", "S1", "D1"):
		conj_mode = input("Incorrect Format; Enter Formula Mode Again: ")
	if conj_mode == "T1": 
		prod_conj = SIG.form_conjunct(T1, conjunct_type = "Product")
		CORRECT = 32
		INCORRECT = 48
	if conj_mode == "T2": 
		prod_conj = SIG.form_conjunct(T2, conjunct_type = "Product")
		CORRECT = 32
		INCORRECT = 48
	if conj_mode == "B1": 
		prod_conj = SIG.form_conjunct(B1, conjunct_type = "Product")
		CORRECT = 32
		INCORRECT = 48
	if conj_mode == "B2": 
		prod_conj = SIG.form_conjunct(B2, conjunct_type = "Product")
		CORRECT = 32
		INCORRECT = 48
	if conj_mode == "S1": 
		prod_conj = SIG.form_conjunct(S1, conjunct_type = "Product")
		CORRECT = 32
		INCORRECT = 48
	# if conj_mode == "D1":
	# 	conj1 = SIG.form_conjunct(T1, conjunct_type="Product")
	# 	conj2 = SIG.form_conjunct(D2, conjunct_type="Product")
	# 	conj3 = SIG.form_conjunct(D3, conjunct_type="Product")
	# 	conj4 = SIG.form_conjunct(D4, conjunct_type="Product")
	# 	prod_conj = stimuli.Disjunct(conj1, conj2, conj3, conj4)
	# 	CORRECT = 32
	# 	INCORRECT = 48
	if LOG_FLAG == True:
		os.makedirs(DIRECTORY + sub_code)
		DIRECTORY += sub_code + "/"
		with open(DIRECTORY + "Sub_resp.csv", "a") as outfile:
			outfile.write("Subject code: " + sub_code + "\n")
			outfile.write("Formula mode: " + conj_mode + "\n")
			outfile.write("Formula List: " + "Product")
			outfile.write("\n")
			outfile.write("\t".join(HEADER))
			outfile.write("\n")

	###########################################################################
	
	# Presentation Settings
	# WIN = visual.Window([1728, 1117], monitor="testMonitor", units="deg", fullscr = True, useRetina = True)
	# WIN = visual.Window([1728, 1117], monitor="testMonitor", units="deg", fullscr = False, useRetina = True, screen = 1)
	WIN = visual.Window([2560, 1440], monitor="testMonitor", units="deg", fullscr = True, useRetina = True, screen = 1)
	global OBJ_LINSPACE, OBJ_HEIGHT
	OBJ_LINSPACE = [-9, 9]
	OBJ_HEIGHT = 4
	OBJ_DICT1 = {
		2: visual.ImageStim(WIN, "rsc/2O3F2D/rcw.png"),
		3: visual.ImageStim(WIN, "rsc/2O3F2D/rcd.png"),
		5: visual.ImageStim(WIN, "rsc/2O3F2D/rtw.png"),
		7: visual.ImageStim(WIN, "rsc/2O3F2D/rtd.png"),
		11: visual.ImageStim(WIN, "rsc/2O3F2D/bcw.png"),
		13: visual.ImageStim(WIN, "rsc/2O3F2D/bcd.png"),
		17: visual.ImageStim(WIN, "rsc/2O3F2D/btw.png"),
		19: visual.ImageStim(WIN, "rsc/2O3F2D/btd.png"),
	}
	OBJ_DICT2 = {
		2: visual.ImageStim(WIN, "rsc/2O3F2D/rcw.png"),
		3: visual.ImageStim(WIN, "rsc/2O3F2D/rcd.png"),
		5: visual.ImageStim(WIN, "rsc/2O3F2D/rtw.png"),
		7: visual.ImageStim(WIN, "rsc/2O3F2D/rtd.png"),
		11: visual.ImageStim(WIN, "rsc/2O3F2D/bcw.png"),
		13: visual.ImageStim(WIN, "rsc/2O3F2D/bcd.png"),
		17: visual.ImageStim(WIN, "rsc/2O3F2D/btw.png"),
		19: visual.ImageStim(WIN, "rsc/2O3F2D/btd.png"),
	}
	OBJ_DICT3 = {
		2: visual.ImageStim(WIN, "rsc/2O3F2D/rcw.png"),
		3: visual.ImageStim(WIN, "rsc/2O3F2D/rcd.png"),
		5: visual.ImageStim(WIN, "rsc/2O3F2D/rtw.png"),
		7: visual.ImageStim(WIN, "rsc/2O3F2D/rtd.png"),
		11: visual.ImageStim(WIN, "rsc/2O3F2D/bcw.png"),
		13: visual.ImageStim(WIN, "rsc/2O3F2D/bcd.png"),
		17: visual.ImageStim(WIN, "rsc/2O3F2D/btw.png"),
		19: visual.ImageStim(WIN, "rsc/2O3F2D/btd.png"),
	}
	OBJ_DICTS = [OBJ_DICT1, OBJ_DICT2, OBJ_DICT3]
	TRIAL_OBJ_DICT = {
		"prompt_msg": visual.TextBox2(WIN, 'Your Prediction: ', pos = [-17, -5], alignment = 'right', letterHeight = 1),
		"true_usr": visual.TextBox2(WIN, 'True', pos = [21, -5], alignment = 'left', color = "Green", letterHeight = 1),
		"fals_usr": visual.TextBox2(WIN, 'False', pos = [21, -5], alignment = 'left', color = "Red", letterHeight = 1),
		"response_msg": visual.TextBox2(WIN, 'Correct Response: ', pos = [-17, -7], alignment = 'right', letterHeight = 1),
		"true_rsp": visual.TextBox2(WIN, 'True', pos = [21, -7], alignment = 'left', color = "Green", letterHeight = 1),
		"fals_rsp": visual.TextBox2(WIN, 'False', pos = [21, -7], alignment = 'left', color = "Red", letterHeight = 1),
		"Correct_sound": sound.Sound("rsc/Correct.wav"),
		"Incorrect_sound": sound.Sound("rsc/Incorrect.wav"),
		"Correct_text": visual.TextBox2(WIN, "Congrats! You prediction is correct!", pos = [0, -10], alignment = 'center', color = "Green", letterHeight = 0.8),
		"Incorrect_text": visual.TextBox2(WIN, "Unfortunately, your prediction is incorrect.", pos = [0, -10], alignment = 'center', color = "Red", letterHeight = 0.8)
	}
	global PROCEED_KEYS
	global ABORT_KEY
	PROCEED_KEYS = ["space", "return"]
	ABORT_KEY = "q"

	###########################################################################

	# define a manual
	ctrl = visual.TextBox2(WIN, "T: True            F: False            1-5: Confidence Rating            0: Back to Stimuli            Space/Enter: Proceed", size = [40,3], alignment = "center", pos = (0,-20), letterHeight = 0.7, fillColor = "black", opacity = 0.5, borderWidth = 0)
	disp_objs = [ctrl]
	
	# Prepare Stimuli
	correct_seq, incorrect_seq = seq_handler(SIG, SIG.sequences, prod_conj, CORRECT, INCORRECT)
	block_seq = correct_seq + incorrect_seq
	block_seq = block_seq

	# Starters
	starter_win(WIN, disp_objs)
	random.shuffle(block_seq)
	repeat_flag = True
	while repeat_flag:
		repeat_flag = test_block(WIN, OBJ_DICTS, TRIAL_OBJ_DICT, block_seq[:4], prod_conj, OBJ_LINSPACE, disp_objs)
	core.wait(1)

	# training block 1
	block_disp_start = visual.TextBox2(WIN, "Now that you are familiar with the presentation software, you are prepared to go through the 80 sequences.", alignment = 'left', letterHeight = 0.8)
	block_disp_end = visual.TextBox2(WIN, "You have gone through all 80 sequences once and think that you are onto something. You can take a short rest before the Corporate urge you to go through them again.", alignment = 'left', letterHeight = 0.8)
	cont_disp0 = visual.TextBox2(WIN, "(This is the start of block 1. Press any key to start.)",pos = (0, -3), alignment = 'center', letterHeight = 0.8)	
	cont_disp1 = visual.TextBox2(WIN, "(This is the end of block 1. Press any key to continue.)",pos = (0, -3), alignment = 'center', letterHeight = 0.8)	
	random.shuffle(block_seq)
	any_cont(WIN, ABORT_KEY, [block_disp_start, cont_disp0])
	core.wait(0.4)
	block_rsp = block(WIN, 1, OBJ_DICTS, TRIAL_OBJ_DICT, block_seq, prod_conj, OBJ_LINSPACE, disp_objs)
	if LOG_FLAG == True:
		with open(DIRECTORY + "Sub_resp.csv", "a") as outfile:
			for rind in range(len(block_rsp)):
				outfile.write("\t".join(block_rsp[rind].astype(str)))
				outfile.write("\n")
	any_cont(WIN, ABORT_KEY, [block_disp_end, cont_disp1])
	core.wait(0.8)

	# training block 2		
	block_disp_start = visual.TextBox2(WIN, "You are now ready to go through the sequences again.", alignment = 'center', letterHeight = 0.8)
	block_disp_end = visual.TextBox2(WIN, "You have gone through the sequences twice and believe that you have more or less gotten down to the sequences that will trigger explosions. You can take a short rest before going through the test block.", alignment = 'left', pos = (0, 2), letterHeight = 0.8)
	# block_disp_end = visual.TextBox2(WIN, "You have gone through the combinations twice and believe that you have more or less gotten down to the combinations that will trigger explosions. However, just when you are about to submit your report …", alignment = 'left', pos = (0, 2), letterHeight = 0.8)
	cont_disp0 = visual.TextBox2(WIN, "(This is the start of block 2. Press any key to start.)",pos = (0, -3), alignment = 'center', letterHeight = 0.8)	
	cont_disp1 = visual.TextBox2(WIN, "(This is the end of block 2. Press any key to continue.)",pos = (0, -3), alignment = 'center', letterHeight = 0.8)	
	random.shuffle(block_seq)
	any_cont(WIN, ABORT_KEY, [block_disp_start, cont_disp0])
	core.wait(0.4)
	block_rsp = block(WIN, 2, OBJ_DICTS, TRIAL_OBJ_DICT, block_seq, prod_conj, OBJ_LINSPACE, disp_objs)
	if LOG_FLAG == True:
		with open(DIRECTORY + "Sub_resp.csv", "a") as outfile:
			for rind in range(len(block_rsp)):
				outfile.write("\t".join(block_rsp[rind].astype(str)))
				outfile.write("\n")
	any_cont(WIN, ABORT_KEY, [block_disp_end, cont_disp1])
	core.wait(0.8)

	# generalization block 1
	block_seq = []
	for seq in SIG.sequences: block_seq.append(seq)
	random.shuffle(block_seq)
	block_disp_start = visual.TextBox2(WIN, "You are now ready to go through the test block. You will only go through 64 sequences this time, and no feedback will be provided.", alignment = 'left', pos = (0, 5), size = [40, None],  letterHeight = 0.8)
	block_disp_end = visual.TextBox2(WIN, "You have gone through the test block and submitted your report to the Corporate.You wondered how you perform, but only the Corporate knows the answer. You gather your thoughts and leave the lab.", alignment = 'left', pos = (0, 2), letterHeight = 0.8)
	cont_disp0 = visual.TextBox2(WIN, "(This is the start of the test block. Press any key to start.)",pos = (0, -3), alignment = 'center', letterHeight = 0.8)	
	cont_disp1 = visual.TextBox2(WIN, "(This is the end of the experiment. Thank you for your participation!)",pos = (0, -3), alignment = 'center', letterHeight = 0.8)
	any_cont(WIN, ABORT_KEY, [block_disp_start, cont_disp0])
	core.wait(0.4)
	block_rsp = block(WIN, "G1", OBJ_DICTS, TRIAL_OBJ_DICT, block_seq, prod_conj, OBJ_LINSPACE, disp_objs, show_truth = False)
	if LOG_FLAG == True:
		with open(DIRECTORY + "Sub_resp.csv", "a") as outfile:
			for rind in range(len(block_rsp)):
				outfile.write("\t".join(block_rsp[rind].astype(str)))
				outfile.write("\n")
	any_cont(WIN, ABORT_KEY, [block_disp_end, cont_disp1])
	core.wait(0.8)

	WIN.close()
	return

def seq_handler(sig, sequences, formula, correct_num, incorrect_num):
	correct_comb = []
	incorrect_comb = []
	for seq in sequences:
		if seq.satisfies(formula): correct_comb.append(seq)
		else: incorrect_comb.append(seq)
	extended_correct = __sq_helper(sig, correct_comb, correct_num)
	extended_incorrect = __sq_helper(sig, incorrect_comb, incorrect_num)
	return extended_correct, extended_incorrect

def __sq_helper(sig, comb, num):
	full_perm = sig.fully_permute_sequences(comb)
	extended_seq = []
	if len(full_perm) < num:
		extended_seq = full_perm
	random.shuffle(comb)
	perm_gen = [seq.permute() for seq in comb]
	curr_ind = 0
	while len(extended_seq) < num:
		if curr_ind >= len(perm_gen): curr_ind = 0
		try:
			extended_seq.append(next(perm_gen[curr_ind]))
			curr_ind += 1
		except StopIteration:
			random.shuffle(comb)
			perm_gen = [seq.permute() for seq in comb]
	return extended_seq

def record_file(directory, name, titles, contents):
	with open(directory + name + ".csv", "a") as outfile:
		outfile.write("\t".join(titles))
		outfile.write("\n")
		for rind in range(len(contents)):
			outfile.write("\t".join(contents[rind].astype(str)))
			outfile.write("\n")
		outfile.write("\n")
	return

def end_win(win, disp_objs = []):
	end_msg0 = visual.TextBox2(win, "This is the end of the experiment.\nThank you for your participation!", pos = (0,2), size = [40, None], letterHeight = 1.5, alignment = "center")
	end_msg1 = visual.TextBox2(win, "Press any key to end", pos = (0,-3), size = [30, None], letterHeight = 0.8, alignment = "center")
	any_cont(win, ABORT_KEY, [end_msg0, end_msg1])
	return

def starter_win(win, disp_objs = []):
	# Signal start of the experiment
	welcome_msg0 = visual.TextBox2(win, "Welcome to the experiment!", pos = (0,2), size = [40, None], letterHeight = 1.5, alignment = "center")
	welcome_msg1 = visual.TextBox2(win, "Press any key to start", pos = (0,-3), size = [30, None], letterHeight = 0.8, alignment = "center")

	any_cont(win, ABORT_KEY, [welcome_msg0, welcome_msg1])
	core.wait(0.4)

	# Control Page
	control_msg0 = visual.TextBox2(win, "Here are your controls for the experiments:", pos = (0,4), size = [31, None], alignment = "left", letterHeight = 1.3)
	ctrl = visual.TextBox2(win, "T: True            F: False            1-5: Confidence Rating            0: Back to Stimuli            Space/Enter: Proceed", size = [40,3], alignment = "center", pos = (0,0), letterHeight = 0.7, fillColor = "black", opacity = 0.5, borderWidth = 0)
	control_msg1 = visual.TextBox2(win, "These control mappings will be kept at the bottom of the screen during the entire experiment. Now, please press space/return to proceed.", pos = (0, -4), size = [31, None], alignment = "left", letterHeight = 0.8)
	spec_cont(win, ABORT_KEY, PROCEED_KEYS, [control_msg0, ctrl, control_msg1])
	core.wait(0.4)

	# Background Messages
	starter_msg0 = visual.TextBox2(win, "Background Story:", pos = (0,7), size = [41, None], alignment = 'left', letterHeight = 1.6)
	starter_msg1 = visual.TextBox2(win,"     You are a newly employed physicist in the Deep Rock Corporate, and are assigned to the F-302 lab to investigate a group of exotic objects. The objects resemble simple geometric shapes and are otherwise unimpressive. However, an earlier report indicates that these objects contain a tremendous amount of energy, but will only release the energy (through an explosion) when they are arranged in a certain combination with other objects. Previous research in the lab had built up an experimental setup that uses disposable robots to arrange the objects, so one can comfortably observe the explosions in your control room. \n\n      There are currently 8 such objects known to the team, and they differ by their colors (red, blue), shapes (circle, triangle), and size (large, small). The Corporate has limited the investigation to a 2-object sequences for now. Your job is to go through all of these sequences and deduce the correct (and minimal) sequences that will always release energy through an explosion.", pos = (0,-2), size = [40, None], alignment = 'left', letterHeight = 0.7)
	spec_cont(win, ABORT_KEY, PROCEED_KEYS, [starter_msg0, starter_msg1] + disp_objs)
	core.wait(0.2)
	
	# More Background notes
	msg1 = visual.TextBox2(win,"Before going into the task, You want to familiarize yourself with the presentation software. Above are all 8 objects. They vary in color and shape.", pos = (0,-14), size = [40, None], alignment = 'left', letterHeight = 0.8)
	obj0 = visual.ImageStim(win, "rsc/2O3F2D/rcl.png", pos = (18, 12))
	obj1 = visual.ImageStim(win, "rsc/2O3F2D/rcs.png", pos = (6, 12))
	obj2 = visual.ImageStim(win, "rsc/2O3F2D/rtl.png", pos = (-6, 12))
	obj3 = visual.ImageStim(win, "rsc/2O3F2D/rts.png", pos = (-18, 12))
	obj4 = visual.ImageStim(win, "rsc/2O3F2D/bcl.png", pos = (18, -3))
	obj5 = visual.ImageStim(win, "rsc/2O3F2D/bcs.png", pos = (6, -3))
	obj6 = visual.ImageStim(win, "rsc/2O3F2D/btl.png", pos = (-6, -3))
	obj7 = visual.ImageStim(win, "rsc/2O3F2D/bts.png", pos = (-18, -3))
	spec_cont(win, ABORT_KEY, PROCEED_KEYS, [msg1, obj0, obj1, obj2, obj3, obj4, obj5, obj6, obj7] + disp_objs)
	core.wait(0.2)

	msg1 = visual.TextBox2(win,"A 2-object sequences is simply two objects that are put into the left and right slot of the alien machine. Above is a sequence represented on the screen. The order of the objects matters: if you exchange the left and right objects to the opposite position the outcome may differ", pos = (0,-7), size = [40, None], alignment = 'left', letterHeight = 0.8)
	obj1 = visual.ImageStim(win, "rsc/2O3F2D/rcl.png", pos = (OBJ_LINSPACE[0],OBJ_HEIGHT))
	obj2 = visual.ImageStim(win, "rsc/2O3F2D/bts.png", pos = (OBJ_LINSPACE[1],OBJ_HEIGHT))
	spec_cont(win, ABORT_KEY, PROCEED_KEYS, [msg1, obj1, obj2] + disp_objs)
	core.wait(0.2)

	msg1 = visual.TextBox2(win,"In this experiment, you will see a number of sequences of objects arranged by the robots, presented one by one. As each sequence is presented, you will be asked by the corporate to predict whether it will explode. You will also be asked to state how confident you are in your prediction. You will then see whether in fact that sequence triggers an explosion, that is, whether your prediction was correct or not. At first you will just have to guess, but on the basis of the feedback you receive, you should gradually be able to learn which sequences trigger an explosion.", pos = (0,1), size = [40, None], alignment = 'left', letterHeight = 0.8)
	spec_cont(win, ABORT_KEY, PROCEED_KEYS, [msg1] + disp_objs)
	core.wait(0.2)

	msg1 = visual.TextBox2(win,"After going through two training blocks, you will encounter the test block. The structure of the test block will be identitical to that of the training blocks, except that you only need to go through 64 sequences and will no longer receive feedback from your choices. The corporate will take your choices in the test block as your report of the experiment.", pos = (0,1), size = [40, None], alignment = 'left', letterHeight = 0.8)
	spec_cont(win, ABORT_KEY, PROCEED_KEYS, [msg1] + disp_objs)
	core.wait(0.2)

	test_msg = visual.TextBox2(win, "Before officially starting the experiment, let's do some practice trials so that you understand the procedure.", pos = (0,0), size = [30, None], letterHeight = 0.8, alignment = "left")
	spec_cont(win, ABORT_KEY, PROCEED_KEYS, [test_msg] + disp_objs)
	core.wait(0.4)
	return

def test_block(win, obj_dicts, trial_obj_dict, sequences, formula, obj_linspace, disp_objs = []):
	test_disp_start = visual.TextBox2(win, "Practice Block", alignment = 'center', letterHeight = 1)
	cont_disp0 = visual.TextBox2(win, "Press any key to start",pos = (0, -3), alignment = 'center', letterHeight = 0.8)
	any_cont(win, ABORT_KEY, [test_disp_start, cont_disp0])

	total_rsp = np.empty((len(sequences), 6), dtype = object)
	for ind, seq in enumerate(sequences):
		curr_disp_objs = disp_objs.copy()
		trial_counter = visual.TextBox2(win, "Trial " + str(ind + 1) + " out of " + str(len(sequences)) + ".", pos = [0,18], alignment = 'center', letterHeight = 0.7)
		curr_disp_objs.append(trial_counter)
		total_rsp[ind] = trial(win, obj_dicts, trial_obj_dict, seq, formula, obj_linspace, curr_disp_objs)
		core.wait(0.4)
	
	test_disp_end = visual.TextBox2(win, "This is the end of the practice block.", pos = [0,0], alignment = 'center', letterHeight = 1)
	cont_disp1 = visual.TextBox2(win, "Press space/return to proceed to Block 1\nPress R to repeat the practice block", pos = (0,-3), letterHeight = 0.8, alignment = 'center')
	test_disp_end.draw()
	cont_disp1.draw()
	win.flip()
	while True:
		allKeys = event.waitKeys()
		break_flag = False
		repeat_flag = False
		for thisKey in allKeys:
			if thisKey == ABORT_KEY: exit("Experiment Aborted")
			if thisKey == "r":
				break_flag = True
				repeat_flag = True
				break
			if thisKey in PROCEED_KEYS:
				break_flag = True
				break
		if break_flag == True: break
	win.flip()
	event.clearEvents()
	core.wait(0.8)
	return repeat_flag

def block(win, block_num, obj_dicts, trial_obj_dict, sequences, formula, obj_linspace, disp_objs = [], show_truth = True):
	total_rsp = np.empty((len(sequences), 7), dtype = object)
	total_rsp[:, 0] = block_num
	for ind, seq in enumerate(sequences):
		curr_disp_objs = disp_objs.copy()
		trial_counter = visual.TextBox2(win, "Trial " + str(ind + 1) + " out of " + str(len(sequences)) + ".", pos = [0,18], alignment = 'center', letterHeight = 0.7)
		curr_disp_objs.append(trial_counter)
		total_rsp[ind, 1:] = trial(win, obj_dicts, trial_obj_dict, seq, formula, obj_linspace, curr_disp_objs, show_truth = show_truth)
		core.wait(0.4)
	return total_rsp

def any_cont(win, abort_key, disp_objs):
	win.flip()
	for obj in disp_objs: obj.draw()
	win.flip()
	while True:
		allKeys = event.waitKeys()
		for thisKey in allKeys: 
			if thisKey == abort_key: exit("Experiment Aborted")
		if len(allKeys) != 0: break
	win.flip()
	event.clearEvents()
	return

def spec_cont(win, abort_key, spec_keys, disp_objs):
	win.flip()
	for obj in disp_objs: obj.draw()
	win.flip()
	while True:
		allKeys = event.waitKeys()
		break_flag = False
		for thisKey in allKeys:
			if thisKey == abort_key: exit("Experiment Aborted")
			if thisKey in spec_keys:
				break_flag = True
				break
		if break_flag == True: break
	win.flip()	
	event.clearEvents()
	return

def trial(win, obj_dicts, trial_obj_dict, Sequence, formula, obj_linspace, disp_objs = [], show_truth = True):
	# Initialize Objects
	# print(Sequence.hierarchical_rep())
	seq_rep = ""
	for obj in Sequence.objects:
		seq_rep += str(obj.id) + "; "
	seq_rep = seq_rep[:-2]

	win_objs = []
	for ind, obj in enumerate(Sequence.objects):
		print(obj.id, end = "; ")
		curr_obj = obj_dicts[ind][obj.id]
		win_objs.append(curr_obj)
	for obj_ind in range(len(win_objs)):
		win_objs[obj_ind].pos = [obj_linspace[obj_ind], 4]
		# win_objs[obj_ind].size = OBJ_SIZE
	print()

	prompt_msg = trial_obj_dict["prompt_msg"]
	true_usr = trial_obj_dict["true_usr"]
	fals_usr = trial_obj_dict["fals_usr"]
	response_msg = trial_obj_dict["response_msg"]
	true_rsp = trial_obj_dict["true_rsp"]
	fals_rsp = trial_obj_dict["fals_rsp"]
	Correct_sound = trial_obj_dict["Correct_sound"]
	Incorrect_sound = trial_obj_dict["Incorrect_sound"]
	Correct_text = trial_obj_dict["Correct_text"]
	Incorrect_text = trial_obj_dict["Incorrect_text"]
	ratingScale = RatingScale(win, low = 0, high = 5, choices = ["\n0\nBack to\nStimuli", "\n1\nJust\nguessing", "\n2\n\n", "\n3\n\n", "\n4\n\n", "\n5\nVery\nConfident"], scale = "How confident are you in your prediction?", markerColor = "orange", pos = (0,0), size = 1.2, stretch = 1.4, textSize = 0.6, acceptPreText = "Your rating: ", acceptKeys = ['return', 'space'], skipKeys = None, showValue = False)
	# choices = ["(0)\nBack to\nStimuli", "(1)\nJust guessing\n", "(2)\n\n", "(3)\n\n", "(4)\n\n", "(5)\nVery Confident\n"]
	ground_truth = Sequence.satisfies(formula)
	
	start_time = time.time()
	trial_rsp = decision_process(win, win_objs, disp_objs, prompt_msg, true_usr, fals_usr)
	end_time = time.time()
	rspt_1 = end_time - start_time
	win.flip()
	win.flip()

	# Confidence Ratings
	core.wait(0.5)
	while ratingScale.noResponse:
		ratingScale.draw()
		for obj in disp_objs: obj.draw()
		win.flip()
	for key in event.getKeys():
			print(key)
	rating = ratingScale.getRating()
	if rating == "\n0\nBack to\nStimuli":
		return trial(win, obj_dicts, trial_obj_dict, Sequence, formula, obj_linspace, disp_objs, show_truth)
	if rating == "\n1\nJust\nguessing": rating = 1
	if rating == "\n2\n\n": rating = 2
	if rating == "\n3\n\n": rating = 3
	if rating == "\n4\n\n": rating = 4
	if rating == "\n5\nVery\nConfident": rating = 5
	print(rating)

	win.flip()
	core.wait(0.1)

	if show_truth == True:
		# Update window
		for obj in win_objs: obj.draw()
		for obj in disp_objs: obj.draw()
		prompt_msg.draw()
		response_msg.draw()
		if ground_truth == True: true_rsp.draw()
		else: fals_rsp.draw()
		if trial_rsp == True:
			true_usr.draw()
		else:
			fals_usr.draw()
		if trial_rsp == ground_truth: 
			Correct_sound.play()
			Correct_text.draw()
		else: 
			Incorrect_sound.play()
			Incorrect_text.draw()
		win.flip(clearBuffer = False)

		# Wait for user response to finish the trial
		start_time = time.time()
		while True:
			allKeys = event.waitKeys()
			break_flag = False
			for thisKey in allKeys:
				if thisKey == "q": exit("Experiment Aborted")
				if thisKey == 'space' or thisKey == 'return':
					break_flag = True
					end_time = time.time()
					break
			if break_flag == True: break
		rspt_2 = end_time - start_time
	else: rspt_2 = -1

	win.flip()
	win.flip()
	event.clearEvents()

	return [seq_rep, trial_rsp, ground_truth, rspt_1, rspt_2, rating]

def decision_process(win, win_objs, disp_objs, prompt_msg, true_usr, fals_usr):
	prev_rsp = None
	# Actual trial
	while True:
		# Drawing
		win.flip()
		for obj in win_objs: obj.draw()
		for obj in disp_objs: obj.draw()
		prompt_msg.draw()
		if prev_rsp is not None: prev_rsp.draw()
		win.flip(clearBuffer = False)

		# Prompt user response
		curr_rsp = None
		while curr_rsp is None:
			allKeys = event.waitKeys()
			for thisKey in allKeys:
				if thisKey == "q": exit("Experiment Aborted")
				if thisKey == "t":
					true_usr.draw()
					prev_rsp = true_usr
					curr_rsp = True
					break
				if thisKey == "f":
					fals_usr.draw()
					prev_rsp = fals_usr
					curr_rsp = False
					break
				if thisKey == 'space' or thisKey == 'return':
					if prev_rsp is not None:
						if prev_rsp == true_usr: return True
						else: return False
		prompt_msg.draw()
		win.flip(clearBuffer = False)


if __name__ == "__main__":
	main()

