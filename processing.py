import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, log_loss
from sklearn.manifold import MDS
from tqdm import tqdm
import random
from scipy import stats
import stimuli
import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import product, combinations, permutations

np.set_printoptions(suppress=True)
pd.set_option('display.max_rows', 1000)
pd.options.mode.chained_assignment = None  # default='warn'

file_name = "Sub_resp.csv"
base_path = "./rsps/"
model_result_path = "./results/"

stimuli_dict = {
	2: "RCL",
	3: "RCS",
	5: "RTL",
	7: "RTS",
	11: "BCL",
	13: "BCS",
	17: "BTL",
	19: "BTS",
}

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

TEST_Conj = [
				("=1", "+0"),
				("=11",),
				("=13",),
			]

reverse_list = {
	19: 2,
	17: 3,
	13: 5,
	11: 7,
	7: 11,
	5: 13,
	3: 17,
	2: 19,
}

List1_G2_convert = {
	(19, 19, 19): (19, 19, 19),
	(19, 13, 19): (19, 19, 19),
	(17, 19, 19): (19, 19, 19),
	(19, 13, 17): (19, 19, 19),
	(2, 2, 2,): (2, 2, 2,),
	(3, 2, 2,): (2, 2, 2,),
	(2, 19, 2): (2, 2, 2,),
	(19, 2, 19): (2, 19, 19),
	(2, 17, 19): (2, 19, 19),
	(17, 17, 2): (2, 19, 19),
	(7, 11, 19): (7, 11, 19),
	(19, 5, 11): (7, 11, 19),
	(11, 13, 7): (7, 11, 19),
	(7, 13, 17): (7, 13, 17),
	(13, 17, 7): (7, 13, 17),
}

list2_G2_convert = {
	(2, 2, 2): (19, 19, 19),
	(2, 5, 2): (19, 19, 19),
	(2, 5, 3): (19, 19, 19),
	(19, 19, 19): (2, 2, 2,),
	(17, 19, 19): (2, 2, 2,),
	(19, 2, 19): (2, 2, 2,),
	(2, 19, 2): (2, 19, 19),
	(19, 3, 2): (2, 19, 19),
	(3, 3, 19): (2, 19, 19),
	(11, 7, 2): (7, 11, 19),
	(2, 13, 7): (7, 11, 19),
	(7, 5, 11): (7, 11, 19),
	(11, 5, 3): (7, 13, 17),
	(5, 3, 11): (7, 13, 17),
}

test_condense = {
	(2,2): (2,2),
	(2,3): (2,2),
	(2,5): (2,5),
	(2,7): (2,5),
	(2,11): (2,11),
	(2,13): (2,11),
	(3,3): (2,2),
	(3,5): (2,5),
	(3,7): (2,5),
	(3,11): (2,11),
	(3,13): (2,11),
	(5,11): (5,11),
	(5,13): (5,11),
	(7,11): (5,11),
	(7,13): (5,11)
}

test_dict = {
	(53, 53): "Negative",
	(2,2): "RC + RC",
	(2,5): "RC + RT",
	(2,11): "RC + BC",
	(5,11): "RT + BC",
}

G2_dict = {
	(19, 19, 19): "Proto Negative",
	(2, 2, 2,): "Proto Positive",
	(2, 19, 19): "Feature Conjunctive",
	(7, 11, 19): "Feature Disjunctive",
	(7, 13, 17): "Feature Presence",
}

COLOR_WHEEL = ["firebrick", "darkorange", "darkcyan", "cyan", "mediumorchid"]

def main():
	
	# Get subject logs
	sublog = {}
	with open("Sub_log.txt", "r") as infile:
		lines = infile.readlines()
		for l in lines:
			curr_l = l.strip("\n")
			curr_l = curr_l.split("\t")
			# print({curr_l[0]:curr_l[1:]})
			sublog.update({curr_l[0]:curr_l[1:]})
	
	all_data = read_data(base_path, file_name)

	# list1_data = all_data[all_data["List"] == "L1"]
	# def list1_transform(seq):
	# 	seq = tuple(seq)
	# 	if seq in List1_G2_convert: 
	# 		return List1_G2_convert[seq]
	# 	else: 
	# 		return seq
	# list1r_data.loc[:,"Seq"] = list1_data["Seq"].apply(list1_transform)
	# list2_data = all_data[all_data["List"] == "L2"]
	# def list2_transform(seq):
	# 	seq = tuple(seq)
	# 	if seq in list2_G2_convert: 
	# 		return list2_G2_convert[seq]
	# 	else: 
	# 		return seq
	# list2_data.loc[:,"Seq"] = list2_data["Seq"].apply(list2_transform)
	# all_data = pd.concat([list1_data, list2_data], axis = 0)
	list1_data = all_data[all_data["List"] == "L1"]
	list2_data = all_data[all_data["List"] == "L2"]
	list2_data.loc[:, "Seq"] = list2_data["Seq"].apply(reverse_stim)
	all_data = pd.concat([list1_data, list2_data], axis = 0)
	# all_data.loc[:, "Seq"] = all_data["Seq"].apply(condense_g2)
	all_data.to_csv("all_data.csv")

	# block_data = all_data[all_data["Formula_Type"] == "T1"]
	# for ind, subname in enumerate(block_data["Subname"].unique()):
	# 	sub_data = block_data[block_data["Subname"] == subname]
	# 	# Get Training Data
	# 	train_data = sub_data.iloc[:160]
	# 	curr_dict = {}
	# 	curr_dict.update({"Training Seq": train_data["Seq"].to_numpy()})
	# 	curr_dict.update({"Training Out": train_data["Truth"].to_numpy()})
	# 	curr_dict.update({"Training Rsp": train_data["Rsp"].to_numpy()})
	# 	# Get Testing Dat
	# 	# test_data = sub_data.iloc[160:192]
	# 	test_data = sub_data[sub_data["Blc"] == "G1"]
	# 	sequences = test_data["Seq"].to_numpy()
	# 	outcomes = test_data["Truth"].to_numpy()
	# 	sub_pred = test_data["Rsp"].to_numpy()
	# 	print(sum(outcomes))
	# exit()
	
	# data, SIG2, SIG3, True_formula = generate_SIG(all_data[all_data["Formula_Type"] == "T1"])
	# for seq in SIG2.sequences: 
	# 	print(seq.satisfies(True_formula))
	# 	print(seq.hierarchical_rep())
	# 	print("================") 


	# # Model-Free Analysis
	# analysis_stream(all_data, "All", "")
	# exit()
	
	# # all_data = all_data[all_data["List"] == "L2"]
	# G2_data = all_data[all_data["Blc"] == "G2"]
	# stimuli_breakdown("T1", G2_data[G2_data["Formula_Type"] == "T1"], G2_dict)
	# stimuli_breakdown("T2", G2_data[G2_data["Formula_Type"] == "T2"], G2_dict)
	# stimuli_breakdown("B1", G2_data[G2_data["Formula_Type"] == "B1"], G2_dict)
	# stimuli_breakdown("B2", G2_data[G2_data["Formula_Type"] == "B2"], G2_dict)
	# stimuli_breakdown("S1", G2_data[G2_data["Formula_Type"] == "S1"], G2_dict)
	# # data = read_file(base_path + "1105_1/", file_name, "test")
	# # print(data)

	# all_data = all_data[all_data["Formula_Type"] == "B2"]
	# all_data = all_data[all_data["Blc"] == "G1"]
	# def test_transform(seq):
	# 	seq = tuple(seq)
	# 	if seq in test_condense:
	# 		return test_condense[seq]
	# 	else:
	# 		return tuple([53, 53])
	# all_data.loc[:, "Seq"] = all_data["Seq"].apply(test_transform)
	# print(all_data["Seq"].unique())
	# stimuli_breakdown("B2", all_data, test_dict)
	# exit()

	# target_data = all_data[all_data["Formula_Type"] == "B2"]
	# target_data = target_data[target_data["Blc"] == "G1"]

	# sub_dist = {}
	# for sub in target_data["Subname"].unique():
	# 	sub_data = target_data[target_data["Subname"] == sub]
	# 	sub_seq = sub_data["Seq"].to_numpy()
	# 	sub_rsp = sub_data["Rsp"].to_numpy()
	# 	sub_tru = sub_data["Truth"].to_numpy()
	# 	sub_ind = np.argsort([np.product(x) for x in sub_seq])
	# 	sub_seq = sub_seq[sub_ind]
	# 	sub_rsp = tuple(sub_rsp[sub_ind])
	# 	sub_tru = sub_tru[sub_ind]
	# 	if sub_rsp in sub_dist: 
	# 		sub_dist[sub_rsp] +=1
	# 	else:
	# 		sub_dist.update({sub_rsp: 1})
	
	# sub_ans = np.array(list(sub_dist.keys()), dtype = int)
	# sub_frq = np.array(list(sub_dist.values()))
	# for ind in range(len(sub_ans)):
	# 	print(sub_frq[ind])
	# 	print(sub_ans[ind])

	# exit()

	# Model-Based Analysis

	# # loading results
	# plot_generalization("T1", ["FB", "F1", "F1-"], prefix = "FM ", sep_width = 0.2, figsize = (10, 15))
	# plot_generalization("T1", ["D3", "S3"], prefix = "OM ", sep_width = 0.2, figsize = (10, 15))
	# plot_generalization("T1", ["FN", "CN"], prefix = "CM ", sep_width = 0.2, figsize = (10, 15))
	# plot_generalization("T2", ["FB", "F1", "F1-"], prefix = "FM ", sep_width = 0.2, figsize = (10, 15))
	# plot_generalization("T2", ["D3", "S3"], prefix = "OM ", sep_width = 0.2, figsize = (10, 15))
	# plot_generalization("T2", ["FN", "CN"], prefix = "CM ", sep_width = 0.2, figsize = (10, 15))
	# plot_generalization("B1", ["FB", "F1", "F1-"], prefix = "FM ", sep_width = 0.2, figsize = (10, 15))
	# plot_generalization("B1", ["D3", "S3"], prefix = "OM ", sep_width = 0.2, figsize = (10, 15))
	# plot_generalization("B1", ["FN", "CN"], prefix = "CM ", sep_width = 0.2, figsize = (10, 15))
	# plot_generalization("B2", ["FB", "F1", "F1-"], prefix = "FM ", sep_width = 0.2, figsize = (10, 15))
	# plot_generalization("B2", ["D3", "S3"], prefix = "OM ", sep_width = 0.2, figsize = (10, 15))
	# plot_generalization("B2", ["FN", "CN"], prefix = "CM ", sep_width = 0.2, figsize = (10, 15))
	# plot_generalization("S1", ["FB", "F1", "F1-"], prefix = "FM ", sep_width = 0.2, figsize = (10, 15))
	# plot_generalization("S1", ["D3", "S3"], prefix = "OM ", sep_width = 0.2, figsize = (10, 15))
	# plot_generalization("S1", ["FN", "CN"], prefix = "CM ", sep_width = 0.2, figsize = (10, 15))
	
	# plot_models_violin("T1", ["FB", "F1", "F1-", "FN", "S2", "D2", "S3", "D3", "CN"], prefix = "Vin ", bar_width = 0.25, figsize = (30, 15))
	# plot_models_violin("T2", ["FB", "F1", "F1-", "FN", "S2", "D2", "S3", "D3", "CN"], prefix = "Vin ", bar_width = 0.25, figsize = (30, 15))
	# plot_models_violin("B1", ["FB", "F1", "F1-", "FN", "S2", "D2", "S3", "D3", "CN"], prefix = "Vin ", bar_width = 0.25, figsize = (30, 15))
	# plot_models_violin("B2", ["FB", "F1", "F1-", "FN", "S2", "D2", "S3", "D3", "CN"], prefix = "Vin ", bar_width = 0.25, figsize = (30, 15))
	# plot_models_violin("S1", ["FB", "F1", "F1-", "FN", "S2", "D2", "S3", "D3",
	# "CN"], prefix = "Vin ", bar_width = 0.25, figsize = (30, 15))
	
	# plot_models_bar("T1", ["FB", "F1", "F1-", "FN", "S2", "D2", "S3", "D3", "CN"], prefix = "Bar ", bar_width = 0.25, figsize = (30, 15))
	# plot_models_bar("T2", ["FB", "F1", "F1-", "FN", "S2", "D2", "S3", "D3", "CN"], prefix = "Bar ", bar_width = 0.25, figsize = (30, 15))
	# plot_models_bar("B1", ["FB", "F1", "F1-", "FN", "S2", "D2", "S3", "D3", "CN"], prefix = "Bar ", bar_width = 0.25, figsize = (30, 15))
	# plot_models_bar("B2", ["FB", "F1", "F1-", "FN", "S2", "D2", "S3", "D3", "CN"], prefix = "Bar ", bar_width = 0.25, figsize = (30, 15))
	# plot_models_bar("S1", ["FB", "F1", "F1-", "FN", "S2", "D2", "S3", "D3", "CN"], prefix = "Bar ", bar_width = 0.25, figsize = (30, 15))	

	combine_models(all_data)
	exit()

	# Generate Hypotheses
	# data, SIG, True_formula = generate_SIG(target_data)

	# all_conjuncts = SIG.generate_feature_conjuncts(1, spec_functions = ["="], spec_numbers = [1])
	# for ind, conj in enumerate(all_conjuncts):
	# 	print(ind)
	# 	print(conj.hierarchical_rep())
	# 	print("================")
	# exit()

	# Feature Hypotheses
	feature_boolean = SIG.generate_feature_conjuncts(0, spec_functions = ["+"], spec_numbers = [1]) + SIG.generate_feature_conjuncts(1, spec_functions = ["+"], spec_numbers = [1]) + SIG.generate_feature_conjuncts(2, spec_functions = ["+"], spec_numbers = [1]) + SIG.generate_feature_conjuncts(3, spec_functions = ["+"], spec_numbers = [1])
	feature_1 = SIG.generate_feature_conjuncts(0, spec_functions = ["=", "+"], spec_numbers = [1,2]) + SIG.generate_feature_conjuncts(1, spec_functions = ["=", "+"], spec_numbers = [1,2])
	feature_n = feature_1 + SIG.generate_feature_conjuncts(2, spec_functions = ["=", "+"], spec_numbers = [1,2])

	hypotheses = [
		feature_boolean,
		feature_1,
		feature_n,
	]
	hypotheses_names = [
		"FB",
		"F1",
		"FN",
	]

	print(hypotheses_names)
	fit_noisy_bayes(data, SIG, True_formula, hypotheses, hypotheses_names)

	# Object Hypotheses
	D0 = SIG.generate_object_conjuncts([0,0])
	D1 = SIG.generate_object_conjuncts([1,0])
	D2S = SIG.generate_object_conjuncts([2,0])
	D2D = SIG.generate_object_conjuncts([1,1])
	D3S = SIG.generate_object_conjuncts([3,0])
	D3D = SIG.generate_object_conjuncts([2,1])
	D4D1 = SIG.generate_object_conjuncts([3,1])
	D4D2 = SIG.generate_object_conjuncts([2,2])
	D5D = SIG.generate_object_conjuncts([3,2])
	D6D = SIG.generate_object_conjuncts([3,3])

	S1_hypotheses = D0 + D1
	S2_hypotheses = D0 + D1 + D2S
	D2_hypotheses = D0 + D1 + D2D
	C2_hypotheses = D0 + D1 + D2S + D2D
	S3_hypotheses = D0 + D1 + D2S + D3S
	D3_hypotheses = D0 + D1 + D2D + D3D
	C3_hypotheses = D0 + D1 + D2S + D2D + D3S + D3D
	CN_hypotheses = D0 + D1 + D2S + D2D + D3S + D3D + D4D1 + D4D2 + D5D + D6D

	hypotheses = [
		S1_hypotheses,
		S2_hypotheses,
		D2_hypotheses,
		C2_hypotheses,
		S3_hypotheses,
		D3_hypotheses,
		C3_hypotheses,
		CN_hypotheses,
	]
	hypotheses_names = [
		"S1",
		"S2",
		"D2",
		"C2",
		"S3",
		"D3",
		"C3",
		"CN",
	]

	print(hypotheses_names)
	fit_noisy_bayes(data, SIG, True_formula, hypotheses, hypotheses_names)
	# data, SIG, True_formula = generate_SIG(T1_data)
	# all_conjuncts = SIG.generate_object_conjuncts([0,0])

	return

################################################################################
# Model-Based Analyses

def plot_generalization(target_condition, target_models, prefix = "", sep_width = 0.1, figsize = (15,15), colors = COLOR_WHEEL):
	model_preds = pd.read_csv(model_result_path + target_condition + "_preds.csv", index_col=0)
	model_preds = model_preds[model_preds["Blc"] == "G2"]
	
	def convert_tuple(seq):
		seq = seq[1:]
		seq = seq[:-1]
		# seq = tuple(int(seq.split(",")))
		seq = tuple(map(int, seq.split(",")))
		return seq

	model_preds.loc[:, "Seq"] = model_preds["Seq"].apply(convert_tuple)
	model_preds.loc[:, "Seq"] = model_preds["Seq"].apply(condense_g2)
	
	avg_res = model_preds.groupby("Seq").mean()
	subs_rsps = avg_res["Subject Pred"].to_numpy()
	true_mask = avg_res["Truth"].to_numpy().astype(bool)

	x_axis = np.arange(len(subs_rsps))
	fig, ax = plt.subplots(1,1, figsize = figsize)
	ax.bar(x_axis[~true_mask], subs_rsps[~true_mask], label = "False Stimuli", color = colors[0])
	ax.bar(x_axis[true_mask], subs_rsps[true_mask], label = "True Stimuli", color = colors[1])

	model_colors = colors[2:]
	rel_pos =  np.linspace(0,sep_width*(len(target_models)-1),len(target_models))
	rel_pos = rel_pos - rel_pos[len(rel_pos)//2]
	if len(target_models)%2 == 0: rel_pos = rel_pos + sep_width/2
	
	for m_ind, model_name in enumerate(target_models):
		curr_avg = model_preds[model_preds["Model"] == model_name].groupby("Seq").mean()["Model Pred"].to_numpy()
		curr_std = model_preds[model_preds["Model"] == model_name].groupby("Seq").std()["Model Pred"].to_numpy()
		ax.errorbar(x_axis + rel_pos[m_ind], curr_avg, curr_std, marker = "^", alpha = 0.8, elinewidth = 2.5, capsize = 4, linewidth = 4, label = model_name, color = model_colors[m_ind])
	
	ax.legend()
	ax.set_xticks(x_axis)
	fig.savefig(prefix + target_condition + ".png", format = "png", dpi = 500, transparent=True)

	# model_preds = model_preds.groupby("Seq").mean()
	# print(model_preds)
	return

def plot_models_bar(target_condition, target_models, prefix = "", bar_width = 0.2, figsize = (15,15), colors = COLOR_WHEEL):
	model_perfs = pd.read_csv(model_result_path + target_condition + "_perfs.csv", index_col=0)
	model_preds = pd.read_csv(model_result_path + target_condition + "_preds.csv", index_col=0)
	model_preds.insert(1, "Squared Difference", np.square(model_preds["Subject Pred"] - model_preds["Model Pred"]))
	models_res = []

	x_axis = np.arange(len(target_models)) + 1
	fig, ax = plt.subplots(1,1, figsize = figsize)
	rel_pos = np.linspace(0,bar_width*(3-1),3)
	rel_pos = rel_pos - rel_pos[len(rel_pos)//2]

	all_data = []
	all_pos = []
	all_colors = []
	for m_ind, model_name in enumerate(target_models):
		tr_perfs = model_perfs[model_name]/196
		tr_subs = tr_perfs.to_numpy()
		te_pred = model_preds[model_preds["Model"] == model_name]
		te_subs = te_pred[te_pred["Blc"] == "G1"].groupby(["Subname"]).mean()["Squared Difference"].to_numpy()
		ge_subs = te_pred[te_pred["Blc"] == "G2"].groupby(["Subname"]).mean()["Squared Difference"].to_numpy()
		ax.bar(x_axis[m_ind]+rel_pos[0], np.average(tr_subs), width = bar_width, label = "Training", alpha = 1, zorder = 1, color = colors[0])
		ax.scatter([x_axis[m_ind]+rel_pos[0]]*len(tr_subs), tr_subs, color = "black", marker = "+", alpha = 0.3, zorder = 2)
		ax.bar(x_axis[m_ind]+rel_pos[1], np.average(te_subs), width = bar_width, label = "Test", alpha = 1, zorder = 1, color = colors[1])
		ax.scatter([x_axis[m_ind]+rel_pos[1]]*len(te_subs), te_subs, color = "black", marker = "+", alpha = 0.3, zorder = 2)
		ax.bar(x_axis[m_ind]+rel_pos[2], np.average(ge_subs), width = bar_width, label = "Generalization", alpha = 1, zorder = 1, color = colors[2])
		ax.scatter([x_axis[m_ind]+rel_pos[2]]*len(ge_subs), ge_subs, color =
		"black", marker = "+", alpha = 0.3, zorder = 2)
		if m_ind == 0: ax.legend()

	ax.set_xticks(x_axis)
	ax.set_xticklabels(target_models)
	fig.savefig(prefix + target_condition + " Results.png", format = "png", dpi = 500, transparent=True)
	return

def plot_models_violin(target_condition, target_models, prefix = "", bar_width = 0.2, figsize = (15,15), colors = COLOR_WHEEL):
	model_perfs = pd.read_csv(model_result_path + target_condition + "_perfs.csv", index_col=0)
	model_preds = pd.read_csv(model_result_path + target_condition + "_preds.csv", index_col=0)
	model_preds.insert(1, "Squared Difference", np.square(model_preds["Subject Pred"] - model_preds["Model Pred"]))
	models_res = []

	x_axis = np.arange(len(target_models)) + 1
	fig, ax = plt.subplots(1,1, figsize = figsize)
	rel_pos = np.linspace(0,bar_width*1.1*(3-1),3)
	rel_pos = rel_pos - rel_pos[len(rel_pos)//2]

	all_data = []
	all_pos = []
	all_colors = []
	for m_ind, model_name in enumerate(target_models):
		tr_perfs = model_perfs[model_name]/196
		tr_subs = tr_perfs.to_numpy()
		te_pred = model_preds[model_preds["Model"] == model_name]
		te_subs = te_pred[te_pred["Blc"] == "G1"].groupby(["Subname"]).mean()["Squared Difference"].to_numpy()
		ge_subs = te_pred[te_pred["Blc"] == "G2"].groupby(["Subname"]).mean()["Squared Difference"].to_numpy()
		all_data.append(tr_subs)
		all_pos.append(x_axis[m_ind]+rel_pos[0])
		all_colors.append(colors[0])
		all_data.append(te_subs)
		all_pos.append(x_axis[m_ind]+rel_pos[1])
		all_colors.append(colors[1])
		all_data.append(ge_subs)
		all_pos.append(x_axis[m_ind]+rel_pos[2])
		all_colors.append(colors[2])
		if m_ind != 0:
			ax.axvline(x_axis[m_ind]-0.5, linestyle = "--", color = "grey")

	parts = ax.violinplot(all_data, positions = all_pos, widths = bar_width, showmeans=False, showextrema=False, showmedians=False)
	for ind, pc in enumerate(parts["bodies"]):
		pc.set_facecolor(colors[ind % 3])
		pc.set_alpha(1)
	quartile1, medians, quartile3 = np.percentile(all_data, [25, 50, 75], axis=1)
	whiskers = np.array([adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(all_data, quartile1, quartile3)])
	whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
	ax.scatter(all_pos, medians, marker='o', color='white', s=30, zorder=3)
	ax.vlines(all_pos, quartile1, quartile3, color='k', linestyle='-', lw=5)
	ax.vlines(all_pos, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

	tr_patch = mpatches.Patch(color = colors[0], label = "Training Results")
	te_patch = mpatches.Patch(color = colors[1], label = "Testing Results")
	ge_patch = mpatches.Patch(color = colors[2], label = "Generalization Results")
	ax.legend(handles = [tr_patch, te_patch, ge_patch])
	ax.set_xticks(x_axis)
	ax.set_xticklabels(target_models)
	fig.savefig(prefix + target_condition + " Results.png", format = "png", dpi = 500, transparent=True)

	return

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def combine_models(target_data):
	# Generate SIG
	Dim_1 = ["Red", "Blue"]
	Dim_2 = ["Circle", "Triangle"]
	Dim_3 = ["Large", "Small"]
	R = 2
	Dim_names = ["fill", "shape", "size"]
	SIG2 = stimuli.Sigma([Dim_1, Dim_2, Dim_3], Dim_names, R, generation_mode="Multiset Combination")
	SIG3 = stimuli.Sigma([Dim_1, Dim_2, Dim_3], Dim_names, R+1, generation_mode="Multiset Combination")

	# test_conj = SIG2.form_conjunct(TEST_Conj, conjunct_type="Sum")
	# print(test_conj.hierarchical_rep())
	# for seq in SIG2.sequences:
	# 	print(test_conj.accepts(seq))
	# 	print(seq.hierarchical_rep())
	# 	print()
	# exit()

	True_formula_dict = {
		"T1": SIG2.form_conjunct(T1, conjunct_type = "Product"),
		"T2": SIG2.form_conjunct(T2, conjunct_type = "Product"),
		"B1": SIG2.form_conjunct(B1, conjunct_type = "Product"),
		"B2": SIG2.form_conjunct(B2, conjunct_type = "Product"),
		"S1": SIG2.form_conjunct(S1, conjunct_type = "Product")
	}

	# Object Hypotheses
	D2S = SIG2.generate_object_conjuncts([2,0])
	D2D = SIG2.generate_object_conjuncts([1,1])
	D3S = SIG2.generate_object_conjuncts([3,0])
	D3D = SIG2.generate_object_conjuncts([2,1])
	D4D1 = SIG2.generate_object_conjuncts([3,1])
	D4D2 = SIG2.generate_object_conjuncts([2,2])
	D5D = SIG2.generate_object_conjuncts([3,2])
	D6D = SIG2.generate_object_conjuncts([3,3])

	print(len(D2S + D2D))
	print(len(D3S + D3D))
	exit()

	OS3 = D2S + D3S
	OD3 = D2D + D3D
	OCN = D4D1 + D4D2 + D5D + D6D

	# Feature Hypotheses
	F1F = SIG2.generate_feature_conjuncts(1, feat_specs = ["=0", "=1", "=2"], relation_specs=["=11", "=13"])
	F2F = SIG2.generate_feature_conjuncts(2, feat_specs = ["=0", "=1", "=2"])
	F3F = SIG2.generate_feature_conjuncts(3, feat_specs = ["=0", "=1", "=2"])
	FBO = SIG2.generate_feature_conjuncts(1, feat_specs = ["+1", "=0"]) + SIG2.generate_feature_conjuncts(2, feat_specs = ["+1", "=0"]) + SIG2.generate_feature_conjuncts(3, feat_specs = ["+1", "=0"])
	# FCN = SIG2.generate_feature_conjuncts(2, feat_specs = ["=0", "=1", "+1", "=2"]) + SIG2.generate_feature_conjuncts(3, feat_specs = ["=0", "=1", "+1", "=2"])
	# FCN = SIG2.generate_feature_conjuncts(2, feat_specs = ["=0", "=1", "+1", "-1", "=2", "+2", "-2"]) + SIG2.generate_feature_conjuncts(3, feat_specs = ["=0", "=1", "+1", "-1", "=2", "+2", "-2"])

	# print(len(F1F))
	# print(len(F2F))
	# print(len(F3F))
	# print(len(F1F + F2F + F3F))
	# # for rule in F1F:
	# # 	print(rule.hierarchical_rep())
	# # 	print()
	# exit()

	models = [OS3, OD3, OCN, F1F, F2F, F3F, FBO]
	model_names = ["OS3", "OD3", "OCN", "F1F", "F2F", "F3F", "FBO"]
	for mn, md in zip(model_names, models):
		print(mn, "\t:", len(md))

	model_perfs = []
	popped_models = []

	while len(models) > 1:
		pop_ind, pop_model, best_perf = reject_model(target_data, models, model_names, SIG2, SIG3, True_formula_dict, None)
		model_perfs.append(best_perf)
		popped_models.append(pop_model)
		models.pop(pop_ind)
		model_names.pop(pop_ind)
		print(pop_model, ":", best_perf)
	print("Complete Evaluation")
	for mn, mp in zip(popped_models, model_perfs):
		print(mn, ":", mp)
	print("Last Model:", model_names[0])
	return

def reject_model(data, models, model_names, SIG2, SIG3, True_formula_dict, pbar):
	HS_performances = []
	
	for ind in range(len(models)):
		curr_HS = []
		curr_model_names = []
		# Get the hypotheses space
		for r_ind, r_model in enumerate(models):
			if r_ind == ind: continue
			curr_HS.extend(models[r_ind])
			curr_model_names.append(model_names[r_ind])
		curr_PHS = stimuli.preloaded_hypothesis_space(curr_HS, SIG2.sequences)
		sub_performances = []
		for sub in data["Subname"].unique():
			sub_data = data[data["Subname"] == sub]
			curr_ftype = data["Formula_Type"].unique()[0]
			curr_groundtruth = True_formula_dict[curr_ftype]
			sub_performances.append(noisy_model_fit_subject(sub_data, SIG2, SIG3, curr_PHS, curr_HS, curr_groundtruth, len(curr_model_names)))
			pbar.update(1)
		HS_performances.append(np.average(sub_performances))


	best_ind = np.argmin(HS_performances)
	print(HS_performances)
	print(best_ind)
	print()
	return best_ind, model_names[best_ind], HS_performances[best_ind]

from multiprocessing import Pool, set_start_method
set_start_method('fork')
from itertools import repeat

def run(args):
	data, SIG2, SIG3, True_formula_dict = closure
	ind, curr_PHS, curr_HS, sub, len_curr_model_names = args
	sub_data = data[data["Subname"] == sub]
	curr_ftype = sub_data["Formula_Type"].unique()[0]
	curr_groundtruth = True_formula_dict[curr_ftype]
	sub_performance = noisy_model_fit_subject(sub_data, SIG2, SIG3, curr_PHS, curr_HS, curr_groundtruth, len_curr_model_names)

	return ind, sub_performance

def reject_model_par(data, models, model_names, SIG2, SIG3, True_formula_dict, pbar):
	global closure
	closure = (data, SIG2, SIG3, True_formula_dict)
	tasks = []
	for ind in range(len(models)):
		curr_HS = []
		curr_model_names = []
		# Get the hypotheses space
		for r_ind, r_model in enumerate(models):
			if r_ind == ind: continue
			curr_HS.extend(models[r_ind])
			curr_model_names.append(model_names[r_ind])
		curr_PHS = stimuli.preloaded_hypothesis_space(curr_HS, SIG2.sequences)

		for sub in data["Subname"].unique():
			tasks.append((ind, curr_PHS, curr_HS, sub, len(curr_model_names)))
	
	with Pool() as p:
		performances = list(tqdm(p.imap_unordered(run, tasks), total=len(tasks)))

	HS_performances = np.zeros(len(models))
	for ind, sub_performance in performances:
		HS_performances[ind] += sub_performance
	HS_performances /= len(data["Subname"].unique())

	best_ind = np.argmin(HS_performances)
	print(HS_performances)
	print(best_ind)
	print()
	return best_ind, model_names[best_ind], HS_performances[best_ind]

reject_model = reject_model_par

def run_models(target_data):
	data, SIG2, SIG3, True_formula = generate_SIG(target_data)
	f_type = data["Formula_Type"].unique()[0]

	# Feature Hypotheses
	feature_boolean = SIG2.generate_feature_conjuncts(0, spec_functions = ["+"], spec_numbers = [1]) + SIG2.generate_feature_conjuncts(1, spec_functions = ["+"], spec_numbers = [1]) + SIG2.generate_feature_conjuncts(2, spec_functions = ["+"], spec_numbers = [1]) + SIG2.generate_feature_conjuncts(3, spec_functions = ["+"], spec_numbers = [1])
	feature_1 = SIG2.generate_feature_conjuncts(0, spec_functions = ["=", "+"], spec_numbers = [1,2]) + SIG2.generate_feature_conjuncts(1, spec_functions = ["=", "+"], spec_numbers = [1,2])
	feature_1_neg = SIG2.generate_feature_conjuncts(0, feat_specs = ["=0", "=1", "+1", "=2", "+2"]) + SIG2.generate_feature_conjuncts(1, feat_specs = ["=0", "=1", "+1", "=2", "+2"])
	feature_n = feature_1_neg + SIG2.generate_feature_conjuncts(2, feat_specs = ["=0", "=1", "+1", "=2", "+2"])

	# Object Hypotheses
	D0 = SIG2.generate_object_conjuncts([0,0])
	D1 = SIG2.generate_object_conjuncts([1,0])
	D2S = SIG2.generate_object_conjuncts([2,0])
	D2D = SIG2.generate_object_conjuncts([1,1])
	D3S = SIG2.generate_object_conjuncts([3,0])
	D3D = SIG2.generate_object_conjuncts([2,1])
	D4D1 = SIG2.generate_object_conjuncts([3,1])
	D4D2 = SIG2.generate_object_conjuncts([2,2])
	D5D = SIG2.generate_object_conjuncts([3,2])
	D6D = SIG2.generate_object_conjuncts([3,3])

	S1_hypotheses = D0 + D1
	S2_hypotheses = D0 + D1 + D2S
	D2_hypotheses = D0 + D1 + D2D
	C2_hypotheses = D0 + D1 + D2S + D2D
	S3_hypotheses = D0 + D1 + D2S + D3S
	D3_hypotheses = D0 + D1 + D2D + D3D
	C3_hypotheses = D0 + D1 + D2S + D2D + D3S + D3D
	CN_hypotheses = D0 + D1 + D2S + D2D + D3S + D3D + D4D1 + D4D2 + D5D + D6D

	hypotheses = [
		feature_boolean,
		feature_1,
		feature_1_neg,
		feature_n,
		S2_hypotheses,
		D2_hypotheses,
		S3_hypotheses,
		D3_hypotheses,
		CN_hypotheses,
	]
	hypotheses_names = [
		"FB",
		"F1",
		"F1-",
		"FN",
		"S2",
		"D2",
		"S3",
		"D3",
		"CN"
	]

	# for conj in feature_1_neg:
	# 	print(conj.hierarchical_rep())
	# 	print(conj.complexity)
	# 	print("")
	# exit()

	print(hypotheses_names)
	Model_Performance, Model_Parameters, Model_Predictions = fit_noisy_bayes(data, SIG2, SIG3, True_formula, hypotheses, hypotheses_names)
	print(Model_Performance)
	Model_Performance.to_csv(model_result_path + f_type + "_perfs" + ".csv")
	Model_Parameters.to_csv(model_result_path + f_type + "_parms" + ".csv")
	Model_Predictions.to_csv(model_result_path + f_type + "_preds" + ".csv")

def generate_SIG(data):
	# Generate SIG
	Dim_1 = ["Red", "Blue"]
	Dim_2 = ["Circle", "Triangle"]
	Dim_3 = ["Large", "Small"]
	R = 2
	Dim_names = ["fill", "shape", "size"]
	SIG2 = stimuli.Sigma([Dim_1, Dim_2, Dim_3], Dim_names, R, generation_mode="Multiset Combination")
	SIG3 = stimuli.Sigma([Dim_1, Dim_2, Dim_3], Dim_names, R+1, generation_mode="Multiset Combination")
	
	# Generate Ground Truth
	f_type = data["Formula_Type"].unique()
	if len(f_type) > 1: raise RuntimeError(len(f_type))
	f_type = f_type[0]
	if f_type == "T1":
		True_formula = SIG2.form_conjunct(T1, conjunct_type = "Product")
	elif f_type == "T2":
		True_formula = SIG2.form_conjunct(T2, conjunct_type = "Product")
	elif f_type == "B1":
		True_formula = SIG2.form_conjunct(B1, conjunct_type = "Product")
	elif f_type == "B2":
		True_formula = SIG2.form_conjunct(B2, conjunct_type = "Product")
	elif f_type == "S1":
		True_formula = SIG2.form_conjunct(S1, conjunct_type = "Product")
	else: raise KeyError(f_type)

	# Double checking if the correct rule is generated
	tr_data = data[np.add(data["Blc"] == "1",data["Blc"] == "2")]
	for i, r in tr_data.iterrows():
		curr_seq = SIG2.get_sequence(r["Seq"])
		truth = curr_seq.satisfies(True_formula)
		rc_truth = r["Truth"]
		if truth != rc_truth: raise RuntimeError()
	
	return data, SIG2, SIG3, True_formula

def fit_noisy_bayes(data, SIG2, SIG3, True_formula, hypotheses, hypotheses_names):
	if len(hypotheses) != len(hypotheses_names): raise RuntimeError
	f_type = data["Formula_Type"].unique()[0]

	model_results = []
	model_predictions = []
	for ind in range(len(hypotheses)):
		curr_res, curr_pred = noisy_model_fit(SIG2, SIG3, hypotheses[ind], data, True_formula, hypotheses_names[ind])
		model_results.append(curr_res)
		model_predictions.append(curr_pred)
	
	all_fits = np.array([res["Model Score"].to_numpy() for res in model_results]).transpose()
	best_fit = np.array([np.argmin(x) for x in all_fits])

	Model_Performance = {
		"Subname": model_results[0]["Subname"],
	}
	Model_Parameters = {
		"Subname": model_results[0]["Subname"],
	}

	for ind, model_name in enumerate(hypotheses_names):
		print("==============================")
		print(hypotheses_names[ind], "Results")
		print(model_results[ind][["Model Score", "Model Alpha", "Model Beta"]].mean())
		print("Best fit:", sum(best_fit == ind), "out of", len(best_fit), "Subjects")
		Model_Performance.update({model_name: model_results[ind]["Model Score"].to_numpy()})
		Model_Parameters.update({model_name + " Alpha": model_results[ind]["Model Alpha"].to_numpy()})
		Model_Parameters.update({model_name + " Beta": model_results[ind]["Model Beta"].to_numpy()})
		# model_results[ind].to_csv(f_type + "_" + hypotheses_names[ind] +
		# ".csv")
	return pd.DataFrame(Model_Performance), pd.DataFrame(Model_Parameters), pd.concat(model_predictions, axis = 0)


# def fit_noisy_bayes(data, SIG, True_formula):
# 	# Test boolean model
# 	all_alphas = []
# 	all_betas = []
	
# 	boolean_hypotheses = []
# 	for conj in SIG.generate_conjuncts(conjunct_type="Bool", subset_type = ">="):
# 		sol_card = len(SIG.satisfies(conj))
# 		# if sol_card > 0: 
# 		conj.preload_sequences(SIG.sequences)
# 		boolean_hypotheses.append(conj)
# 	bool_res, bool_pred = noisy_model_fit(SIG, boolean_hypotheses, data, True_formula)
# 	all_alphas.append(bool_res["Model Alpha"].mean())
# 	all_betas.append(bool_res["Model Beta"].mean())

# 	summation_hypotheses = []
# 	for conj in SIG.generate_conjuncts(conjunct_type="Sum", subset_type = "=="):
# 		sol_card = len(SIG.satisfies(conj))
# 		# if sol_card > 0: 
# 		conj.preload_sequences(SIG.sequences)
# 		summation_hypotheses.append(conj)
# 	sum_res, sum_pred = noisy_model_fit(SIG, summation_hypotheses, data, True_formula)
# 	all_alphas.append(sum_res["Model Alpha"].mean())
# 	all_betas.append(sum_res["Model Beta"].mean())

# 	product_hypotheses = []
# 	for conj in SIG.generate_conjuncts(conjunct_type="Product", subset_type = ">="):
# 		sol_card = len(SIG.satisfies(conj))
# 		# if sol_card > 0: 
# 		conj.preload_sequences(SIG.sequences)
# 		product_hypotheses.append(conj)
# 	prod_res, prod_pred = noisy_model_fit(SIG, product_hypotheses, data, True_formula)
# 	all_alphas.append(prod_res["Model Alpha"].mean())
# 	all_betas.append(prod_res["Model Beta"].mean())


# 	bool_pred = condense_stimuli(bool_pred)
# 	bool_pred = bool_pred.groupby("Seq", as_index = False).mean()
# 	bool_pred["Order"] = bool_pred["Seq"].apply(np.prod)
# 	bool_pred = bool_pred.sort_values(["Order"]).reset_index(drop=True)
	
# 	sum_pred = condense_stimuli(sum_pred)
# 	sum_pred = sum_pred.groupby("Seq", as_index = False).mean()
# 	sum_pred["Order"] = sum_pred["Seq"].apply(np.prod)
# 	sum_pred = sum_pred.sort_values(["Order"]).reset_index(drop=True)

# 	prod_pred = condense_stimuli(prod_pred)
# 	prod_pred = prod_pred.groupby("Seq", as_index = False).mean()
# 	prod_pred["Order"] = prod_pred["Seq"].apply(np.prod)
# 	prod_pred = prod_pred.sort_values(["Order"]).reset_index(drop=True)

# 	group_map = bool_pred["Truth"].to_numpy().astype(bool)
# 	bool_pred_rearranged = bool_pred["Model Pred"]
# 	sum_pred_rearranged = sum_pred["Model Pred"]
# 	prod_pred_rearranged = prod_pred["Model Pred"]
# 	bool_acc, sum_acc, prod_acc = [], [], []
# 	for ind in range(len(group_map)):
# 		if group_map[ind] == True:
# 			bool_acc.append(bool_pred_rearranged[ind])
# 			sum_acc.append(sum_pred_rearranged[ind])
# 			prod_acc.append(prod_pred_rearranged[ind])
# 		else:
# 			bool_acc.append(1-bool_pred_rearranged[ind])
# 			sum_acc.append(1-sum_pred_rearranged[ind])
# 			prod_acc.append(1-prod_pred_rearranged[ind])
# 	data = pd.DataFrame({"Bool Accuracy": bool_acc, "Sum Accuracy": sum_acc, "Prod Accuracy": prod_acc}).to_numpy()
# 	group_names = {"True Positive Rate": True, "False Negative Rate": False}
# 	xticklabels = bool_pred["Seq"].to_numpy()
# 	for ind in range(len(xticklabels)):
# 		xticklabels[ind] = UC_dict[xticklabels[ind][0]] + " + " + UC_dict[xticklabels[ind][1]]
# 	ylabels = ["Boolean Model", "Summation Model", "Product Model"]

# 	stacked_barplot(data, group_map, group_names, xticklabels, ylabels, "Averaged Prediction", "./")
# 	return bool_res, sum_res, prod_res, all_alphas, all_betas

def noisy_model_fit_subject(curr_data, SIG2, SIG3, PHS, hypotheses_space, True_formula, n_params):
	sub = curr_data["Subname"].unique()[0]
	alpha_range = np.linspace(0, 1, 21)
	beta_range = np.linspace(0, 1, 21)
	complex_prior = stimuli.gamma_prior(hypotheses_space)

	alpha_preds = []
	tr_data = curr_data[np.add(curr_data["Blc"] == "1",curr_data["Blc"] == "2")]
	te_data = curr_data[curr_data["Blc"] == "G1"]
	ge_data = curr_data[curr_data["Blc"] == "G2"]

	alpha_bacc = []
	alpha_betas = []
	sub_models = []
	for alpha in alpha_range:
		model_accs = []
		beta_preds = []
		beta_models = []
		for beta in beta_range:	
			bayes_model = stimuli.Bayesian_model(hypotheses_space, priors = complex_prior, alpha = alpha, beta = beta, preloaded_hypothesis_space = PHS)
			pred_res = []
			sub_res = []
			sub_truth = []
			sub_conf = []
			for i, r in tr_data.iterrows():
				curr_seq = SIG2.get_sequence(r["Seq"])
				truth = curr_seq.satisfies(True_formula)
				sub_conf.append(r["Cnfdnc"])
				sub_res.append(r["Rsp"])
				pred_res.append(bayes_model.prediction(curr_seq))
				sub_truth.append(truth)
				bayes_model.update(curr_seq, truth)
			model_acc = n_params*np.log(len(tr_data)) + 2*log_loss(sub_res, pred_res)
			# model_acc = np.sum(np.square(np.array(sub_res) -
			# np.array(pred_res)))
			model_accs.append(model_acc)
			beta_preds.append(pred_res)
			beta_models.append(bayes_model)
		best_ind = np.argmin(model_accs)
		alpha_betas.append(beta_range[best_ind])
		alpha_bacc.append(model_accs[best_ind])
		alpha_preds.append(beta_preds[best_ind])
		sub_models.append(beta_models[best_ind])
	best_ind = np.argmin(alpha_bacc)
	best_alpha = alpha_range[best_ind]
	best_beta = alpha_betas[best_ind]
	best_bacc = alpha_bacc[best_ind]
	best_pred = alpha_preds[best_ind]
	best_model = sub_models[best_ind]
	
	return best_bacc

	# Model_Res = {
	# 	"Subname": [],
	# 	"Model": [],
	# 	"Blc": [],
	# 	"Order": [],
	# 	"Seq": [],
	# 	"Truth": [],
	# 	"Model Pred": [],
	# 	"Subject Pred": [],
	# }

	# for i,r in te_data.iterrows():
	# 	curr_seq = SIG2.get_sequence(r["Seq"])
	# 	model_pred = best_model.prediction(curr_seq)
	# 	Model_Res["Subname"].append(sub)
	# 	Model_Res["Model"].append(model_name)
	# 	Model_Res["Order"].append(np.prod(r["Seq"]))
	# 	Model_Res["Seq"].append(r["Seq"])
	# 	Model_Res["Truth"].append(r["Truth"])
	# 	Model_Res["Blc"].append("G1")
	# 	Model_Res["Model Pred"].append(model_pred)
	# 	Model_Res["Subject Pred"].append(r["Rsp"])
	# best_model.preloaded_hypothesis_space = None
	# for i, r in ge_data.iterrows():
	# 	curr_seq = SIG3.get_sequence(r["Seq"])
	# 	model_pred = best_model.prediction(curr_seq)
	# 	Model_Res["Subname"].append(sub)
	# 	Model_Res["Model"].append(model_name)
	# 	Model_Res["Order"].append(np.prod(r["Seq"]))
	# 	Model_Res["Seq"].append(r["Seq"])
	# 	Model_Res["Truth"].append(r["Truth"])
	# 	Model_Res["Blc"].append("G2")
	# 	Model_Res["Model Pred"].append(model_pred)
	# 	Model_Res["Subject Pred"].append(r["Rsp"])

	# return pd.DataFrame({"Subname": [sub], "Model Score": [best_bacc], "Model
	# Alpha": [best_alpha], "Model Beta": [best_beta]}), Model_Res

def noisy_model_fit(SIG2, SIG3, hypotheses_space, data, True_formula, model_name):
	alpha_range = np.linspace(0, 1, 21)
	beta_range = np.linspace(0, 1, 21)
	PHS = stimuli.preloaded_hypothesis_space(hypotheses_space, SIG2.sequences)

	Model_Res = {
		"Subname": [],
		"Model": [],
		"Blc": [],
		"Order": [],
		"Seq": [],
		"Truth": [],
		"Model Pred": [],
		"Subject Pred": [],
	}

	pbar = tqdm(total = len(data["Subname"].unique()))
	# For each subject
	sub_names, sub_alpha, sub_beta, sub_score,  = [], [], [], []
	counter = 0
	for sub in data["Subname"].unique():
		# counter += 1
		# if counter > 2: break
		alpha_preds = []
		sub_names.append(sub)
		curr_data = data[data["Subname"] == sub]
		tr_data = curr_data[np.add(curr_data["Blc"] == "1",curr_data["Blc"] == "2")]
		te_data = curr_data[curr_data["Blc"] == "G1"]
		ge_data = curr_data[curr_data["Blc"] == "G2"]
		prob_linspace = np.linspace(0, 1, 9)

		alpha_bacc = []
		alpha_betas = []
		sub_models = []
		for alpha in alpha_range:
			model_accs = []
			beta_preds = []
			beta_models = []
			for beta in beta_range:	
				complex_prior = stimuli.gamma_prior(hypotheses_space)
				bayes_model = stimuli.Bayesian_model(hypotheses_space, priors = complex_prior, alpha = alpha, beta = beta, preloaded_hypothesis_space = PHS)
				pred_res = []
				sub_res = []
				sub_truth = []
				sub_prob = []
				sub_conf = []
				test_sub_res = []
				test_pred_res = []
				for i, r in tr_data.iterrows():
					curr_seq = SIG2.get_sequence(r["Seq"])
					truth = curr_seq.satisfies(True_formula)
					sub_conf.append(r["Cnfdnc"])
					# if truth:
					# 	sub_prob.append(prob_linspace[3 + r["Cnfdnc"]])
					# else:
					# 	sub_prob.append(prob_linspace[5 - r["Cnfdnc"]])
					sub_res.append(r["Rsp"])
					# pred, prior = bayes_model.prediction(curr_seq)
					# pred_res.append(pred.dot(prior))
					pred_res.append(bayes_model.prediction(curr_seq))
					sub_truth.append(truth)
					if r["Blc"] != "G1":
						bayes_model.update(curr_seq, truth)
					else:
						test_sub_res.append(r["Rsp"])
						test_pred_res.append(pred_res[-1])
				model_acc = np.sum(np.square(np.array(sub_res) -
				np.array(pred_res)))
				# model_acc = np.sum(np.square(np.array(test_sub_res) - np.array(test_pred_res)))
				model_accs.append(model_acc)
				beta_preds.append(pred_res)
				beta_models.append(bayes_model)
			best_ind = np.argmin(model_accs)
			alpha_betas.append(beta_range[best_ind])
			alpha_bacc.append(model_accs[best_ind])
			alpha_preds.append(beta_preds[best_ind])
			sub_models.append(beta_models[best_ind])
		best_ind = np.argmin(alpha_bacc)
		sub_alpha.append(alpha_range[best_ind])
		sub_beta.append(alpha_betas[best_ind])
		sub_score.append(alpha_bacc[best_ind])
		best_pred = alpha_preds[best_ind]
		best_model = sub_models[best_ind]
		for i,r in te_data.iterrows():
			curr_seq = SIG2.get_sequence(r["Seq"])
			model_pred = best_model.prediction(curr_seq)
			Model_Res["Subname"].append(sub)
			Model_Res["Model"].append(model_name)
			Model_Res["Order"].append(np.prod(r["Seq"]))
			Model_Res["Seq"].append(r["Seq"])
			Model_Res["Truth"].append(r["Truth"])
			Model_Res["Blc"].append("G1")
			Model_Res["Model Pred"].append(model_pred)
			Model_Res["Subject Pred"].append(r["Rsp"])
		best_model.preloaded_hypothesis_space = None
		for i, r in ge_data.iterrows():
			curr_seq = SIG3.get_sequence(r["Seq"])
			model_pred = best_model.prediction(curr_seq)
			Model_Res["Subname"].append(sub)
			Model_Res["Model"].append(model_name)
			Model_Res["Order"].append(np.prod(r["Seq"]))
			Model_Res["Seq"].append(r["Seq"])
			Model_Res["Truth"].append(r["Truth"])
			Model_Res["Blc"].append("G2")
			Model_Res["Model Pred"].append(model_pred)
			Model_Res["Subject Pred"].append(r["Rsp"])

		# Model_Res["Subname"]+=list(curr_data["Subname"])
		# Model_Res["Seq"]+=list(curr_data["Seq"])
		# Model_Res["Truth"]+=list(curr_data["Truth"])
		# Model_Res["Blc"]+=list(curr_data["Blc"])
		# Model_Res["Model Pred"]+=best_pred
		# Model_Res["Order"]+=list(curr_data["Seq"].apply(np.prod))
		pbar.update(1)
	# print(pd.DataFrame({"Sub name": sub_names, "Model Accuracy": sub_score,
	# "Model Inverse Learning Rate": sub_rate}))
	Model_Res = pd.DataFrame(Model_Res)
	return pd.DataFrame({"Subname": sub_names, "Model Score": sub_score, "Model Alpha": sub_alpha, "Model Beta": sub_beta}), Model_Res


################################################################################
# Model-Free Analyses

def analysis_stream(data, condition_name, prefix = ""):
	# Categorical Analysis
	curr_data = data[data["Formula_Type"] == "T1"]
	T1_statistics = process_df(curr_data)
	curr_data = data[data["Formula_Type"] == "T2"]
	T2_statistics = process_df(curr_data)
	curr_data = data[data["Formula_Type"] == "B1"]
	B1_statistics = process_df(curr_data)
	curr_data = data[data["Formula_Type"] == "B2"]
	B2_statistics = process_df(curr_data)
	curr_data = data[data["Formula_Type"] == "S1"]
	S1_statistics = process_df(curr_data)

	block_contrast_ttest("T1", T1_statistics)
	block_contrast_ttest("T2", T2_statistics)
	block_contrast_ttest("B1", B1_statistics)
	block_contrast_ttest("B2", B2_statistics)
	block_contrast_ttest("S1", S1_statistics)
	cond_contrast_ttest("B2",  ["S1", "C1", "S2", "C2", "P"], [T1_statistics["2"], T2_statistics["2"], B1_statistics["2"], B2_statistics["2"], S1_statistics["2"]])
	cond_contrast_ttest("T1", ["S1", "C1", "S2", "C2", "P"], [T1_statistics["G1"], T2_statistics["G1"], B1_statistics["G1"], B2_statistics["G1"], S1_statistics["G1"]])
	cond_contrast_ttest("T2", ["S1", "C1", "S2", "C2", "P"], [T1_statistics["G2"], T2_statistics["G2"], B1_statistics["G2"], B2_statistics["G2"], S1_statistics["G2"]])

	bar_plot_2B(condition_name + " Condition Performance", ["S1", "C1", "S2", "C2", "P"], [T1_statistics, T2_statistics, B1_statistics, B2_statistics, S1_statistics], prefix = prefix, figsize=(15, 10))
	bar_plot_2B_threshold(condition_name + " Condition Performance Threshold", ["S1", "C1", "S2", "C2", "P"], [T1_statistics, T2_statistics, B1_statistics, B2_statistics, S1_statistics], prefix = prefix, figsize=(15, 10))

	plt.close()
	return T1_statistics, T2_statistics, B1_statistics, B2_statistics, S1_statistics

def stimuli_breakdown(name, data, stimuli_dict, prefix = ""):
	curr_res = extract_stimuli_stats(data)
	plot_stimuli_stats(name + " Stimuli Breakdown", curr_res, stimuli_dict, prefix = prefix)

def process_df(data, window_size = 20, step_size = 20):
	# Getting Statistics
	blc_statistics = dict({})
	all_subs = data["Subname"].unique()
	for blc in data["Blc"].unique():
		blc_statistics.update({blc:dict({})})
		CRR = []
		RT1 = []
		RT2 = []
		Cfn = []
		CRR_win = []
		RT1_win = []
		RT2_win = []
		Cfn_win = []
		for sub in all_subs:
			# Getting aggregated statistics
			sub_rsp = data[data["Subname"] == sub]
			blc_rsp = sub_rsp[sub_rsp["Blc"] == blc]
			curr_res = extract_stats(blc_rsp)
			CRR.append(curr_res[0])
			RT1.append(curr_res[1])
			RT2.append(curr_res[2])
			Cfn.append(curr_res[3])
			# Getting windowed statistics
			if len(blc_rsp) < window_size:
				sw_ind = [np.arange(len(blc_rsp))]
			else:
				sw_ind = list(sliding_window(np.arange(len(blc_rsp)), window_size=window_size, step_size=step_size))
			sub_CRR_win = []
			sub_RT1_win = []
			sub_RT2_win = []
			sub_Cfn_win = []
			for iw in sw_ind:
				win_blc = blc_rsp.iloc[iw]
				curr_res = extract_stats(win_blc)
				sub_CRR_win.append(curr_res[0])
				sub_RT1_win.append(curr_res[1])
				sub_RT2_win.append(curr_res[2])
				sub_Cfn_win.append(curr_res[3])
			CRR_win.append(sub_CRR_win)
			RT1_win.append(sub_RT1_win)
			RT2_win.append(sub_RT2_win)
			Cfn_win.append(sub_Cfn_win)
		# Here, the first element is always subject
		blc_statistics[blc].update({"CRR": np.array(CRR)})
		blc_statistics[blc].update({"RT1": np.array(RT1)})
		blc_statistics[blc].update({"RT2": np.array(RT2)})
		blc_statistics[blc].update({"Cfn": np.array(Cfn)})
		blc_statistics[blc].update({"CRR_win": np.array(CRR_win)})
		blc_statistics[blc].update({"RT1_win": np.array(RT1_win)})
		blc_statistics[blc].update({"RT2_win": np.array(RT2_win)})
		blc_statistics[blc].update({"Cfn_win": np.array(Cfn_win)})

	return blc_statistics

def extract_stats(data):
	CRR = []
	RT1 = []
	RT2 = []
	Cfn = []
	difference = np.square(data["Rsp"].astype(int) - data["Truth"].astype(int))
	CRR= 1-sum(difference)/len(difference)
	RT1= np.average(data["Rsp_t1"])
	RT2= np.average(data["Rsp_t2"])
	Cfn= np.average(data["Cnfdnc"])
	# Cfn= 0.5 + np.average((data["Cnfdnc"] - 1)/8)
	return CRR, RT1, RT2, Cfn

def sliding_window(idx, window_size, step_size = 1):
	if len(idx) <= window_size:
		return idx
	for i in range(0,len(idx)- window_size + 1, step_size):
		yield idx[i:i+window_size]

def extract_stimuli_stats(data):
	data = data.copy()
	data["Seq"] = data["Seq"].apply(lambda x: tuple(np.sort(x)))
	all_stimuli = data["Seq"].unique()

	stimuli_dict = {
		"Seq": [],
		"Truth": [],
		"Freq": [],
		"CRR": [],
		"RT1": [],
		"RT2": [],
		"Cfn": [],
		"Odr": [],
	}

	for stim in all_stimuli:
		stimuli_dict["Seq"].append(stim)
		stimuli_dict["Odr"].append(np.product(stim))
		curr_data = data[data["Seq"] == stim]
		truth_val = curr_data["Truth"].unique()
		if len(truth_val) > 1: 
			raise RuntimeError(stim)
		stimuli_dict["Truth"].append(truth_val[0])
		stimuli_dict["Freq"].append(len(curr_data[curr_data["Subname"] == curr_data["Subname"].unique()[0]]))
		res = extract_stats(curr_data)
		stimuli_dict["CRR"].append(res[0])
		stimuli_dict["RT1"].append(res[1])
		stimuli_dict["RT2"].append(res[2])
		stimuli_dict["Cfn"].append(res[3])

	stim_res = pd.DataFrame(stimuli_dict)
	# return stim_res.sort_values(["Truth", "CRR"])
	return stim_res.sort_values(["Odr"])

def block_contrast_ttest(cond_name, statistics):
	CRR_b1 = statistics["1"]["CRR"]
	CRR_b2 = statistics["2"]["CRR"]
	CRR_t1 = statistics["G1"]["CRR"]
	CRR_t2 = statistics["G2"]["CRR"]
	pval1 = np.around(stats.ttest_rel(CRR_b1, CRR_b2).pvalue, decimals = 5)
	pval2 = np.around(stats.ttest_rel(CRR_b2, CRR_t1).pvalue, decimals = 5)
	pval3 = np.around(stats.ttest_rel(CRR_b2, CRR_t2).pvalue, decimals = 5)
	pval4 = np.around(stats.ttest_rel(CRR_t1, CRR_t2).pvalue, decimals = 5)

	osp_pval1 = np.around(stats.ttest_1samp(CRR_b1, 0.5).pvalue, decimals=5)
	osp_pval2 = np.around(stats.ttest_1samp(CRR_b2, 0.5).pvalue, decimals=5)
	osp_pval3 = np.around(stats.ttest_1samp(CRR_t1, 0.5).pvalue, decimals=5)
	osp_pval4 = np.around(stats.ttest_1samp(CRR_t2, 0.5).pvalue, decimals=5)
	print("-------------------------------------------------------------------")
	print("Performance Overview for", cond_name)
	print("\t-- B1 Acc:", np.around(np.mean(CRR_b1), decimals = 3), "+-", np.around(np.std(CRR_b1), decimals = 3), " Acc > 0.8:", sum(CRR_b1 >= 0.8), "|",  len(CRR_b1))
	print("\t-- B2 Acc:", np.around(np.mean(CRR_b2), decimals = 3), "+-", np.around(np.std(CRR_b2), decimals = 3), " Acc > 0.8:", sum(CRR_b2 >= 0.8), "|",  len(CRR_b2))
	print("\t-- T1 Acc:", np.around(np.mean(CRR_t1), decimals = 3), "+-", np.around(np.std(CRR_t1), decimals = 3), " Acc > 0.8:", sum(CRR_t1 >= 0.8), "|",  len(CRR_t1))
	print("\t-- T2 Acc:", np.around(np.mean(CRR_t2), decimals = 3), "+-", np.around(np.std(CRR_t2), decimals = 3), " Acc > 0.8:", sum(CRR_t2 >= 0.8), "|",  len(CRR_t2))
	print("Block Performances for", cond_name)
	print("\t-- 1 Sample T-test Result B1", osp_pval1, end = " ")
	if osp_pval1 < 0.05: print("*", end="")
	print()
	print("\t-- 1 Sample T-test Result B2", osp_pval2, end = " ")
	if osp_pval2 < 0.05: print("*", end="")
	print()
	print("\t-- 1 Sample T-test Result T1", osp_pval3, end = " ")
	if osp_pval3 < 0.05: print("*", end="")
	print()
	print("\t-- 1 Sample T-test Result T2", osp_pval4, end = " ")
	if osp_pval4 < 0.05: print("*", end="")
	print()

	print("Block Contrast for", cond_name)
	print("\t-- Related T-test Result B1 vs. B2:", pval1, end = " ")
	if pval1 < 0.05: print("*", end="")
	print()
	print("\t-- Related T-test Result B2 vs. T1:", pval2, end = " ")
	if pval2 < 0.05: print("*", end="")
	print()
	print("\t-- Related T-test Result B2 vs. T2:", pval3, end = " ")
	if pval3 < 0.05: print("*", end="")
	print()
	print("\t-- Related T-test Result T1 vs. T2:", pval4, end = " ")
	if pval4 < 0.05: print("*", end="")
	print()
	return

def cond_contrast_ttest(block_name, cond_names, statistics):
	if len(cond_names) != len(statistics): raise RuntimeError()
	print("-------------------------------------------------------------------")
	print("Condition Contrast for", block_name)
	for curr_contrast in combinations(np.arange(len(cond_names)), 2):
		ind1 = curr_contrast[0]
		ind2 = curr_contrast[1]
		curr_pval = stats.ttest_ind(statistics[ind1]["CRR"], statistics[ind2]["CRR"]).pvalue
		curr_pval = np.around(curr_pval, decimals = 5)
		print("\t-- Independent T-test Result for", cond_names[ind1],"vs.", cond_names[ind2],":",curr_pval, end=" ")
		if curr_pval < 0.05: print("*", end="")
		print()
	return

def bar_plot_2B(title, labels, statistics, bar_width = 0.1, figsize = (15,15), prefix = "", colors = COLOR_WHEEL):
	all_CRR_subs = []
	all_CRRs = []
	for stat in statistics:
		all_CRR_subs.append(np.array([stat["1"]["CRR"],
			    				  	  stat["2"]["CRR"],
							      	  stat["G1"]["CRR"],
							      	  stat["G2"]["CRR"]]).transpose())
		all_CRRs.append(np.average(all_CRR_subs[-1], axis = 0))
	rel_pos = np.linspace(0,bar_width*(len(statistics)-1),len(statistics))
	rel_pos = rel_pos - rel_pos[len(rel_pos)//2]
	if len(statistics)%2 == 0: rel_pos = rel_pos + bar_width/2

	x_axis = np.array([1,2,3,4])
	block_labels = np.array(["Training 1", "Training 2", "Testing", "Generalization"])
	fig, ax = plt.subplots(1,1, figsize = figsize)

	for b_ind in range(len(all_CRRs)):
		ax.bar(x_axis + rel_pos[b_ind], all_CRRs[b_ind], width = bar_width, label = labels[b_ind], alpha = 1, zorder = 1, color = colors[b_ind])
		ax.scatter(np.sort([1,2,3,4]*all_CRR_subs[b_ind].shape[0]) + rel_pos[b_ind], all_CRR_subs[b_ind].transpose().flatten(), color = 'black', marker = "+", alpha = 0.3, zorder=2)
	ax.axhline(y = 0.5, color = "black", linestyle = "--", label = "chance", zorder = 2)
	ax.legend(loc = "upper left")
	ax.set_xticks(x_axis)
	ax.set_xticklabels(block_labels)
	ax.set_ylim([0.0,1.05])
	fig.savefig(prefix + title + ".png", format = "png", dpi = 500, transparent=True)
	return

def bar_plot_2B_threshold(title, labels, statistics, bar_width = 0.1, figsize = (15,15), prefix = "", colors = COLOR_WHEEL):
	rel_pos = np.linspace(0,bar_width*(len(statistics)-1),len(statistics))
	rel_pos = rel_pos - rel_pos[len(rel_pos)//2]
	if len(statistics)%2 == 0: rel_pos = rel_pos + bar_width/2

	all_threshold_rates = []
	for stat in statistics:
		cur_CRR = np.array([stat["1"]["CRR"],
			    			stat["2"]["CRR"],
							stat["G1"]["CRR"],
							stat["G2"]["CRR"]]).transpose()
		threshold_rate = np.sum(cur_CRR  > 0.8, axis = 0) / cur_CRR.shape[0]
		all_threshold_rates.append(threshold_rate)

	x_axis = np.array([1,2,3,4])
	block_labels = np.array(["Training 1", "Training 2", "Testing", "Generalization"])
	fig, ax = plt.subplots(1,1, figsize = figsize)

	for b_ind in range(len(all_threshold_rates)):
		ax.bar(x_axis + rel_pos[b_ind], all_threshold_rates[b_ind], width = bar_width, label = labels[b_ind], alpha = 1, zorder = 1, color = colors[b_ind])
	ax.legend(loc = "upper left")
	ax.set_xticks(x_axis)
	ax.set_xticklabels(block_labels)
	ax.set_ylim([0.0,1.05])
	fig.savefig(prefix + title + ".png", format = "png", dpi = 500, transparent=True)

	# for stat in statistics:
	# 	all_CRR_subs.append(np.array([stat["1"]["CRR"],
	# 		    				  	  stat["2"]["CRR"],
	# 						      	  stat["G1"]["CRR"],
	# 						      	  stat["G2"]["CRR"]]).transpose())
	# 	all_CRRs.append(np.average(all_CRR_subs[-1], axis = 0))

	# x_axis = np.array([1,2,3,4])
	# block_labels = np.array(["Training 1", "Training 2", "Testing", "Generalization"])
	# fig, ax = plt.subplots(1,1, figsize = figsize)

	# for b_ind in range(len(all_CRRs)):
	# 	ax.bar(x_axis + rel_pos[b_ind], all_CRRs[b_ind], width = bar_width, label = labels[b_ind], alpha = 1, zorder = 1, color = colors[b_ind])
	# 	ax.scatter(np.sort([1,2,3,4]*all_CRR_subs[b_ind].shape[0]) + rel_pos[b_ind], all_CRR_subs[b_ind].transpose().flatten(), color = 'black', marker = "+", alpha = 0.3, zorder=2)
	# ax.axhline(y = 0.5, color = "black", linestyle = "--", label = "chance", zorder = 2)
	# ax.legend()
	# ax.set_xticks(x_axis)
	# ax.set_xticklabels(block_labels)
	# ax.set_ylim([0.2,1.05])
	# fig.savefig(prefix + title + ".png", format = "png", dpi = 500, transparent=True)
	return

def plot_stimuli_stats(title, stim_res, stim_dict, prefix = ""):
	x_axis = np.arange(len(stim_res))
	labels = stim_res["Seq"].to_numpy()
	for ind in range(len(labels)):
		# labels[ind] = stim_dict[labels[ind][0]] + " + " +
		# stim_dict[labels[ind][1]] + " | " + str(stim_res["Freq"].iloc[ind])
		labels[ind] = stim_dict[labels[ind]]
	True_mask = stim_res["Truth"].to_numpy()
	# fig, ax = plt.subplots(1,1, figsize = (24,10))
	fig, ax = plt.subplots(1,1, figsize = (12, 9))
	ax.bar(x_axis[~True_mask] - 0.2, stim_res["CRR"].to_numpy()[~True_mask], label = "False Negative Rate", color = "turquoise", width = 0.4)
	ax.bar(x_axis[True_mask] - 0.2, stim_res["CRR"].to_numpy()[True_mask], label = "True Positive Rate", color = "royalblue", width = 0.4)
	ax.bar(x_axis + 0.2, stim_res["Cfn"].to_numpy(), label = "Confidence", color = "orange", width = 0.4)
	ax.axhline(y = 0.5, color = "black", linestyle = "--", label = "chance", zorder = 2)
	ax.legend()
	ax.set_xticks(x_axis)
	ax.set_xticklabels(labels, rotation = 0)
	ax.set_ylim([0,1.05])
	fig.savefig(prefix + title + ".png", format = "png", dpi = 500, transparent=True)
	plt.clf()
	return


################################################################################
# Read Functions

def read_data(base_path, file_name):
	# Get subject logs
	sublog = {}
	subcheck = {}
	with open("Sub_log.txt", "r") as infile:
		lines = infile.readlines()
		for l in lines:
			curr_l = l.strip("\n")
			curr_l = curr_l.split("\t")
			sublog.update({curr_l[0]:curr_l[1:]})
			subcheck.update({curr_l[0]:curr_l[1]})

	# Get the data
	all_data = []
	for sub_dir in os.listdir(base_path):
		if sub_dir != ".DS_Store":
			sub_list = sublog[sub_dir][1]
			sub_data = read_file(base_path+sub_dir+"/", file_name, sub_list)
			if sub_data["Subname"].unique()[0] != sub_dir:
				raise RuntimeError("Mismatch between sublog and subject data at", sub_dir, "with subname", sub_data["Subname"].unique()[0])
			all_data.append(sub_data)
	all_data = pd.concat(all_data, axis = 0)

	# for sub in all_data["Subname"].unique():
	# 	curr_data = all_data[all_data["Subname"] == sub]
	# 	print(sub, len(curr_data))

	return all_data

def read_file(rsp_path, file_name, l_type):	
	with open(rsp_path + file_name, "r") as infile:
		lines = infile.readlines()
		sub_name = lines[0].strip("\n")[14:]
		f_type = lines[1].strip("\n")[14:]
	header = 3
	data = pd.read_csv(rsp_path + file_name, sep = "\t", header = header)
	data["Seq"] = data["Seq"].apply(lambda x: list(map(int, x.split(";"))))
	data.insert(0, "Formula_Type", [f_type]*len(data))
	data.insert(0, "List", [l_type]*len(data))
	data.insert(0, "Subname", [sub_name]*len(data))
	return data

def reverse_stim(seq):
	for ind in range(len(seq)):
		seq[ind] = reverse_list[seq[ind]]
	return seq

def condense_g2(seq):
	seq = tuple(seq)
	if seq in List1_G2_convert: 
		return List1_G2_convert[seq]
	else: 
		return seq

if __name__ == "__main__":
	main()