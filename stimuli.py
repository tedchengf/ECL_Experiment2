import numpy as np
import random
from scipy.stats import gamma
from scipy.special import logsumexp
from itertools import product, combinations, permutations

class Object():
	def __init__(self, identity, encoding):
		self.id = None
		self.encoding = None
		self.abrv_encoding = None

		self.__initialize(identity, encoding)

	def contains(self, f_state):
		if type(f_state) is tuple: return self.__contains__(f_state)
		res = np.zeros((len(f_state), len(self.encoding)), dtype = bool)
		for ind, f_s in enumerate(f_state): 
			res[ind, :] = self.__contains__(f_s)
		return res

	def summarize(self):
		res_dict = {}
		for enc in self.encoding:
			if enc in res_dict: res_dict[enc] += 1
			else: res_dict.update({enc: 1})
		return res_dict

	def hierarchical_rep(self):
		repr_str = "Obj " + str(self.id) + "\n"
		repr_str += hierarchical_rep(self.summarize(), base_indent = "  ") + "\n"
		return repr_str[:-1]

	def __getitem__(self, key):
		res = []
		for f in self.encoding:
			if f[0] == key: res.append(f)
		return res

	def __len__(self):
		return len(self.encoding)

	def __iter__(self):
		return self.encoding.__iter__()

	def __contains__(self, key):
		if type(key) is not tuple: raise TypeError("The input key must be a tuple")
		# empty query is automatically true
		if len(key) == 0: return True
		res = np.zeros(len(self.encoding), dtype = bool)
		for ind, enc in enumerate(self.encoding):
			res[ind] = key == enc
		return res

	def __eq__(self, other):
		if isinstance(other, Object) == False: return False
		return self.id == other.id	
	def __ne__(self, other):
		return not self.__eq__(other)
	def __hash__(self):
		return hash(str(self.id))

	def __str__(self):
		return "Obj " + str(self.id) + " : " + self.encoding.__str__()

	def __repr__(self):
		return "Obj " + str(self.id)

	def __initialize(self, identity, encoding):
		if type(identity) is not int: raise TypeError("identity must be an int")
		self.id = identity
		self.encoding = encoding
		self.abrv_encoding = tuple((enc[1] for enc in self.encoding))

class Sequence():
	def __init__(self, objects):
		self.objects = None

		self.__cid = None
		self.__pid = None

		self.__initialize(objects)

	def cequal(self, other):
		if isinstance(other, Sequence) == False: raise TypeError("The input variable must be an instance of Sequence")
		return self.cid == other.cid

	def pequal(self, other):
		if isinstance(other, Sequence) == False: raise TypeError("The input variable must be an instance of Sequence")
		diff_arr = np.subtract(self.__pid, other.pid)
		# return not np.any(diff_arr)
		return diff_arr == 0

	def permute(self, unique = True):
		if unique == True:
			unique_elem = list(set(self.objects))
			counts = [self.objects.count(obj) for obj in unique_elem]
			perms = list(unique_perm(unique_elem, counts, [0]*len(self.objects), len(self.objects) - 1, constructor = Sequence))
			index = np.random.permutation(np.arange(len(perms)))
			for ind in index: yield perms[ind]
		elif unique == False:
			perms = list(permutations(self.objects, len(self.objects)))
			index = np.random.permutation(np.arange(len(perms)))
			for ind in index: yield perms[ind]
		# return [permutations[ind] for ind in index]
		# return unique_perm(unique_elem, counts, [0]*len(self.objects),
		# len(self.objects) - 1, constructor = Sequence)

	def shuffle(self):
		obj_indices = np.arange(len(self.objects))
		random.shuffle(obj_indices)
		return Sequence([self.objects[i] for i in obj_indices])

	def summarize(self, level = "Feature"):
		if level == "Object":
			res_dict = {}
			for obj in self.objects:
				if obj in res_dict: res_dict[obj] += 1
				else: res_dict.update({obj: 1})
			return res_dict
		if level == "Feature":
			sum_dicts = []
			for obj in self.objects: sum_dicts.append(obj.summarize())
			return sum_dicts
		raise ValueError("Unaccepted level for summarize")

	def satisfies(self, conjunct):
		if isinstance(conjunct, Conjunct) != True and isinstance(conjunct, Disjunct) != True and isinstance(conjunct, Sequential_Conjunct) != True: raise TypeError("The input Conjunct must be an instance of Conjunct or Disjunct")
		return conjunct.accepts(self)

		if conjunct.conjunct_type == "Sum":
			f_dicts = self.summarize(level = "Feature")
			f_sum = merge_dicts(*f_dicts)
			return Sub_Func(conjunct.query, f_sum)
		if conjunct.conjunct_type == "Product":
			s_list = self.summarize(level = "Feature")
			formula = conjunct.query.copy()
			return recur_verify_obj(formula, s_list, Sub_Func)

	def hierarchical_rep(self, level = "Object"):
		repr_str = self.__repr__() + "\n"
		if level == "Feature":
			obj_dict = self.summarize(level = "Feature")
			obj_dict = merge_dicts(*obj_dict)
			repr_str += hierarchical_rep(obj_dict, base_indent = "  ")
			return repr_str
		if level == "Object":
			for obj in self.objects:
				repr_str += "  Obj " + str(obj.id) + "\n"
				repr_str += hierarchical_rep(obj.summarize(), base_indent = "    ") + "\n"
			return repr_str[:-1]

	def __eq__(self, other):
		if isinstance(other, Sequence) == False: return False
		return self.pid == other.pid
	def __ne__(self, other):
		return not self.__eq__(other)
	def __hash__(self):
		return hash(self.pid)

	def __getattr__(self, name):
		if name == "cid": return self.__cid
		if name == "pid": return tuple(self.__pid)
		if name == "encodings":
			encodings = []
			for obj in self.objects:
				encodings.append(obj.encoding)
			return tuple(encodings)
		if name == "abrv_encodings":
			abrv_encodings = []
			for obj in self.objects:
				abrv_encodings.append(obj.abrv_encoding)
			return tuple(abrv_encodings)			
		raise AttributeError("Unknown attribute " + name)

	def __getitem__(self, key):
		return self.objects[key]

	def __len__(self):
		return len(self.objects)

	def __iter__(self):
		return self.objects.__iter__()

	def __contains__(self, q_obj):
		if isinstance(q_obj, Object) == False: raise TypeError("The input variable must be an instance of Object")
		for obj in self.objects:
			match_flag = obj == q_obj
			if match_flag == True: return match_flag
		return False

	def __str__(self):
		repr_str = "Seq ("
		for obj in self.objects: repr_str += obj.__repr__() + ", "
		repr_str = repr_str[:-2] + ")"
		return repr_str

	def __repr__(self):
		return "Seq " + str(self.pid)

	def __initialize(self, objects):
		if len(objects) == 0: raise RuntimeError("The sequence must have a least 1 object")
		self.objects = tuple(objects)
		self.__pid = np.zeros(len(objects), dtype = int)
		for ind, obj in enumerate(self.objects):
			self.__pid[ind] = obj.id
		self.__cid = np.prod(self.__pid)
		return

class Sigma():
	def __init__(self, features, feature_names, r, generation_mode = "Multiset Combination"): 
		self.features = None
		self.cardinality = None
		self.objects = None
		self.sequences = None
		self.r = None
		self.generation_mode = None

		self.__feature_dict = None
		self.__pid_to_seq = None
		self.__oid_to_obj = None

		self.__initialize(features, feature_names, r, generation_mode)

	def summarize(self):
		all_states = []
		for f in self: all_states += f
		return all_states.copy()

	def satisfies(self, conjunct, return_non_solutions = False):
		solutions = []
		non_solutions = []
		for seq in self.sequences:
			if seq.satisfies(conjunct): 
				solutions.append(seq)
			else:
				non_solutions.append(seq)
		if return_non_solutions == False:
			return solutions
		else:
			return solutions, non_solutions

	# Here the implementation is only about combination
	def get_sequence(self, cid):
		cid = tuple(sorted(cid))
		return self.__pid_to_seq[cid]

	def generate_sequence(self, pid):
		objects = []
		for oid in pid: objects.append(self.__oid_to_obj[oid])
		return Sequence(objects)

	# def form_conjunct(self, abrv_defn, conjunct_type = "Sum", subset_type = ">="):
	# 	if conjunct_type == "Bool": conjunct_type = "Sum"
	# 	if conjunct_type == "Sum":
	# 		conjunct_dict = {}
	# 		for f_ind, f_spec in enumerate(abrv_defn):
	# 			curr_feature = self[f_ind]
	# 			for fs_ind, fs_spec in enumerate(f_spec):
	# 				if fs_spec > 0: conjunct_dict.update({curr_feature[fs_ind]: fs_spec})
	# 		return Conjunct(conjunct_dict, subset_type)
	# 	if conjunct_type == "Product":
	# 		conjunct_list = []
	# 		for obj_config in abrv_defn:
	# 			conjunct_dict = {}
	# 			for f_ind, f_spec in enumerate(obj_config):
	# 				curr_feature = self[f_ind]
	# 				for fs_ind, fs_spec in enumerate(f_spec):
	# 					if fs_spec > 0: conjunct_dict.update({curr_feature[fs_ind]: fs_spec})
	# 			conjunct_list.append(conjunct_dict)
	# 		return Conjunct(conjunct_list, subset_type)

	def form_conjunct(self, abrv_defn, conjunct_type = "Sum"):
		if conjunct_type == "Bool": conjunct_type = "Sum"
		if conjunct_type == "Sum":
			conjunct_dict = {}
			for f_ind, f_spec in enumerate(abrv_defn):
				curr_feature = self[f_ind]
				# Relation term
				if len(f_spec) == 1:
					curr_key = (self.features[f_ind], "Relation")
					conjunct_dict.update({curr_key: f_spec[0]})
				else:
					for fs_ind, fs_spec in enumerate(f_spec):
						conjunct_dict.update({curr_feature[fs_ind]: fs_spec})
			return Conjunct(conjunct_dict)
		if conjunct_type == "Product":
			conjunct_list = []
			for obj_config in abrv_defn:
				conjunct_dict = {}
				for f_ind, f_spec in enumerate(obj_config):
					curr_feature = self[f_ind]
					for fs_ind, fs_spec in enumerate(f_spec):
						conjunct_dict.update({curr_feature[fs_ind]: fs_spec})
				conjunct_list.append(conjunct_dict)
			return Conjunct(conjunct_list)
		if conjunct_type == "Seq":
			conjunct_list = []
			for obj_config in abrv_defn:
				conjunct_dict = {}
				for f_ind, f_spec in enumerate(obj_config):
					curr_feature = self[f_ind]
					for fs_ind, fs_spec in enumerate(f_spec):
						conjunct_dict.update({curr_feature[fs_ind]: fs_spec})
				conjunct_list.append(conjunct_dict)
			return Sequential_Conjunct(conjunct_list)

	def generate_conjuncts(self, conjunct_type = "Product"):
		all_conjuncts = []
		if conjunct_type == "Bool":
			full_config_gen = self.__generate_object_conj(2)
		elif conjunct_type == "Sum":
			full_config_gen = self.__generate_object_conj(self.r + 1)
		elif conjunct_type == "Product":
			obj_conj = list(self.__generate_object_conj(2))
			full_config_gen = product(*[obj_conj]*self.r)
		else: raise KeyError
		for full_config in full_config_gen:
			all_conjuncts.append(self.form_conjunct(full_config, conjunct_type = conjunct_type))
		return all_conjuncts

	# def generate_conjuncts(self, conjunct_type = "Product", subset_type = ">="):
	# 	all_conjuncts = []
	# 	if conjunct_type == "Bool":
	# 		full_config_gen = self.__generate_object_conj(2)
	# 	elif conjunct_type == "Sum":
	# 		full_config_gen = self.__generate_object_conj(self.r + 1)
	# 	elif conjunct_type == "Product":
	# 		obj_conj = list(self.__generate_object_conj(2))
	# 		full_config_gen = product(*[obj_conj]*self.r)
	# 	else: raise KeyError
	# 	for full_config in full_config_gen:
	# 		all_conjuncts.append(self.form_conjunct(full_config, conjunct_type = conjunct_type, subset_type = subset_type))
	# 	return all_conjuncts

	# assumes binary feature, and assumes 2 max objects
	def generate_object_conjuncts(self, config, verify_conjuncts = True):
		if len(config) == 0 or len(config) > self.r: raise RuntimeError("Invalid number of objects")
		if sum(config) > len(self.features)*self.r: raise RuntimeError("Invalid number of features")
		for f_config in config:
			if f_config > len(self.features): raise RuntimeError("Invalid number of features")
		
		base_config = [("+0", "+0")]
		all_obj_cfg = []
		for f_config in config:
			curr_obj_cfg = []
			for feature_selection in combinations(np.arange(len(self.features)), f_config):
				product_list = []
				for ind in range(len(self.features)):
					if ind in feature_selection:
						product_list.append(list(permutations(["=1", "=0"], 2)))
					else:
						product_list.append(base_config)
				curr_obj_cfg += list(product(*product_list))
			all_obj_cfg.append(curr_obj_cfg)

		all_conjuncts = []
		# Take out repetitions
		if len(config) == 2 and config[0] == config[1]:
			valid_inds = multiset_comb(len(all_obj_cfg[0]), len(config))
			for ind_pair in valid_inds:
				all_conjuncts.append(self.form_conjunct([all_obj_cfg[0][ind_pair[0]], all_obj_cfg[1][ind_pair[1]]], conjunct_type="Product"))
		else:
			full_config = product(*all_obj_cfg)
			for cfg in full_config: 
				all_conjuncts.append(self.form_conjunct(cfg, conjunct_type="Product"))
		if verify_conjuncts == False:
			return all_conjuncts
		else:
			valid_conjuncts = []
			for conj in all_conjuncts:
				if len(self.satisfies(conj)) > 0: valid_conjuncts.append(conj)
			if len(valid_conjuncts) < 1: raise RuntimeError("No valid conjunct Found")
			return valid_conjuncts

	def generate_feature_conjuncts(self, num_features, spec_functions = ["-", "=", "+"], spec_numbers = None, feat_specs = None, relation_specs = None, verify_conjuncts = True):
		if spec_numbers is not None: spec_numbers = np.array(spec_numbers).astype(str)
		base_config = []
		if spec_numbers is None: spec_numbers = np.arange(self.r + 1)[1:].astype(str)
		
		if feat_specs is None:
			feat_specs = [''.join(x) for x in product(spec_functions, spec_numbers)]
		base_spec = [("+0", "+0")]
		single_specs_A = list(product(["+0"], feat_specs))
		single_specs_B = list(product(feat_specs, ["+0"]))
		complex_specs = list(product(feat_specs, feat_specs))
		all_specs = single_specs_A + single_specs_B + complex_specs
		## accepts x and y, and + and =
		if relation_specs is not None:
			all_rel = []
			for rspec in relation_specs: all_rel.append(tuple([rspec]))
		all_specs += all_rel

		all_configs = []
		for feature_selection in combinations(np.arange(len(self.features)), num_features):
			product_list = []
			for ind in range(len(self.features)):
				if ind in feature_selection: product_list.append(all_specs)
				else: product_list.append(base_spec)
			all_configs += list(product(*product_list))

		all_conjuncts = []
		for cfg in all_configs: 
			all_conjuncts.append(self.form_conjunct(cfg))
		if verify_conjuncts == False:
			return all_conjuncts
		else:
			valid_conjuncts = []
			for conj in all_conjuncts:
				if len(self.satisfies(conj)) > 0: valid_conjuncts.append(conj)
			if len(valid_conjuncts) < 1: raise RuntimeError("No valid conjunct Found")
			return valid_conjuncts

	def __generate_object_conj(self, max_config):
		f_configs = []
		for feature in self:
			curr_configrange = list(range(max_config))
			all_config = [curr_configrange]*len(feature)
			config_gen = product(*all_config)
			curr_f = []
			for config in config_gen:
				if sum(config) <= self.r: curr_f.append(config)
			f_configs.append(curr_f)
		full_config_gen = product(*f_configs)
		return full_config_gen

	def fully_permute_sequences(self, sequences = None):
		if sequences is None: sequences = self.sequences
		full_seq = []
		for seq in sequences:
			for seq_perm in seq.permute(): full_seq.append(seq_perm)
		random.shuffle(full_seq)
		return full_seq

	def __generate_config_space(self, specificity):
		fs_configs = []
		for feature_i, feature_spec in enumerate(specificity):
			curr_config = []
			curr_feature = self[feature_i]
			if feature_spec > len(curr_feature): raise RuntimeError("The given specificity at pos " + str(feature_i) + " exceeds the cardinality of the corresponding feature.")
			if feature_spec > 0:
				for s_config in product(*np.repeat(np.arange(feature_spec + 1).reshape(1, -1), len(curr_feature), axis = 0)):
					if sum(s_config) == feature_spec: 
						curr_config.append(s_config)
			fs_configs.append(curr_config)
		return product(*fs_configs)

	def __getitem__(self, key):
		if key is None: return np.array([])
		if type(key) is str:
			try:
				return self.__feature_dict[key]
			except KeyError:
				raise KeyError("The key " + key + " is undefined.")
		if type(key) is int:
			return self.__feature_dict[self.features[key]]
		raise TypeError("The input key must be a str")

	def __len__(self):
		return len(self.features)

	def __iter__(self):
		for key in self.features:
			yield self.__feature_dict[key]

	def __contains__(self, key):
		return key in self.__feature_dict

	def __initialize(self, features, feature_names, r, generation_mode):
		self.features = []
		self.cardinality = []
		self.__feature_dict = {}
		self.r = r
		self.generation_mode = generation_mode
		self.__pid_to_seq = dict({})
		self.__oid_to_obj = dict({})

		for f, f_name in zip(features, feature_names):
			self.features.append(f_name)
			f_list = []
			for f_s in f:
				f_list.append((f_name, f_s))
			self.__feature_dict.update({f_name: f_list})
			self.cardinality.append(len(f))

		self.features = tuple(self.features)
		self.cardinality = tuple(self.cardinality)

		self.objects = self.__generate_objects()
		oids = [obj.id for obj in self.objects]
		for ind in range(len(oids)): self.__oid_to_obj.update({oids[ind]: self.objects[ind]})
		self.sequences = self.__populate(r, generation_mode)
		for seq in self.sequences: self.__pid_to_seq.update({seq.pid: seq})
		
	def __populate(self, r, generation_mode = "Multiset Combination"):
		sequences = []
		if generation_mode == "Combination":
			seqs = combinations(self.objects, r)
			for seq in seqs:
				sequences.append(Sequence(seq))
			return sequences
		if generation_mode == "Permutation":
			seqs = permutations(self.objects, r)
			for seq in seqs:
				sequences.append(Sequence(seq))
			return sequences
		if generation_mode == "Multiset Combination":
			for indices in multiset_comb(len(self.objects), r):
				seq = tuple(self.objects[i] for i in indices)
				sequences.append(Sequence(seq))
			return sequences
		if generation_mode == "Multiset Permutation":
			for indices in multiset_perm(len(self.objects), r):
				seq = tuple(self.objects[i] for i in indices)
				sequences.append(Sequence(seq))
			return sequences
		
		raise ValueError("Unaccepted generation_mode for populate")

	def __generate_objects(self):
		feature_arrs = []
		for f in self: feature_arrs.append(f)
		obj_arrs = []
		primes = get_primes()
		for encoding in product(*feature_arrs):
			obj_id = next(primes)
			obj_arrs.append(Object(obj_id, encoding))
		return obj_arrs

# class Conjunct():
# 	def __init__(self, query, subset_type = ">="):
# 		self.query = None
# 		self.conjunct_type = None
# 		self.subset_type = None
# 		self.complexity = None

# 		self.preloaded_sequences = None
# 		self.___initialize(query, subset_type)

# 	def ___initialize(self, query, subset_type):
# 		if type(query) is dict:
# 			self.conjunct_type = "Sum"
# 		elif type(query) is list:
# 			for q_dict in query:
# 				if type(q_dict) is not dict:
# 					raise TypeError("If the query is a list, its element must be dicts")
# 			self.conjunct_type = "Product"
# 		else:
# 			raise TypeError("The input query must be a dict or a list of dicts")
# 		self.query = query
# 		if subset_type not in (">=", "=="): raise ValueError("The input subset_type must be either '==' or '>='.")
# 		self.subset_type = subset_type 

# 		self.complexity = 1
# 		if self.conjunct_type == "Product":
# 			for obj_dict in self.query:
# 				if len(obj_dict) > 0:
# 					self.complexity += len(obj_dict) + 1
# 		else:
# 			self.complexity += len(self.query)
# 			# for k in self.query:
# 			# 	self.complexity += self.query[k]

# 	def preload_sequences(self, sequences):
# 		self.preloaded_sequences = dict({})
# 		for seq in sequences:
# 			self.preloaded_sequences.update({seq.cid: seq.satisfies(self)})
# 		return

# 	def hierarchical_rep(self, feature_order = None):
# 		if self.conjunct_type == "Sum":
# 			repr_str = "Sum Conjunct\n"
# 			repr_str += hierarchical_rep(self.query, base_indent = "  ", key_order = feature_order)
# 			return repr_str
# 		if self.conjunct_type == "Product":
# 			repr_str = "Product Conjunct\n"
# 			for ind, obj_dict in enumerate(self.query):
# 				repr_str += "  Object " + str(ind) + "\n"
# 				repr_str += hierarchical_rep(obj_dict, base_indent = "    ", key_order = feature_order) + "\n"
# 			return repr_str[:-1]

# 	def accepts(self, sequence):
# 		if isinstance(sequence, Sequence) == False: raise TypeError("The input must be an instance of Sequence")

# 		if self.preloaded_sequences is not None:
# 			return self.preloaded_sequences[sequence.cid]

# 		# if self.subset_type == "==": Sub_Func = subset_dicts_eq
# 		# elif self.subset_type == ">=": Sub_Func = subset_dicts_geq
# 		# else: raise ValueError("The subset_type of the current conjunct must
# 		# be either '==' or '>='.")
# 		Sub_Func = subset_dicts_general

# 		if self.conjunct_type == "Sum":
# 			f_dicts = sequence.summarize(level = "Feature")
# 			f_sum = merge_dicts(*f_dicts)
# 			return Sub_Func(self.query, f_sum)
		
# 		if self.conjunct_type == "Product":
# 			s_list = sequence.summarize(level = "Feature")
# 			formula = self.query.copy()
# 			return recur_verify_obj(formula, s_list, Sub_Func)

# 	def flattened_rep(self, feature_names, feature_lists):
# 		if self.conjunct_type == "Sum":
# 			return self.__flatten_dict(self.query,feature_names,feature_lists)
# 		if self.conjunct_type == "Product":
# 			final_rep = []
# 			for obj_dict in self.query: final_rep.append(self.__flatten_dict(obj_dict, feature_names, feature_lists))
# 			return tuple(final_rep)

# 	def __flatten_dict(self, query, feature_names, feature_lists):
# 		final_rep = []
# 		for f in feature_lists:
# 			final_rep.append([0]*len(f))
# 		if len(query) == 0: return tuple(final_rep)
# 		for key in query:
# 			f_pos = feature_names.index(key[0])
# 			fs_pos = feature_lists[f_pos].index(key[1])
# 			final_rep[f_pos][fs_pos] = query[key]
# 		return tuple(tuple(rep) for rep in final_rep)

# 	def __len__(self):
# 		return len(self.query)

# 	def __str__(self):
# 		return self.conjunct_type + " Conjunct " + self.query.__str__()

# 	def __repr__(self):
# 		return self.conjunct_type + " Conjunct " + self.query.__repr__() 

class Sequential_Conjunct():
	def __init__(self, query):
		self.query = None
		self.conjunct_type = None
		self.complexity = None

		self.preloaded_sequences = None
		self.___initialize(query)

	def ___initialize(self, query):
		if type(query) is not list: raise TypeError("The query must be a list of dicts")
		for q_dict in query:
			if type(q_dict) is not dict: raise TypeError("The query must be a list of dicts")
		self.conjunct_type = "Seq Object"
		self.query = query

		self.complexity = 1
		for obj_dict in self.query:
			obj_complex = 0
			for spec in obj_dict.values():
				if spec != "+0" and spec != "=0": obj_complex += 1
			if obj_complex != 0: obj_complex += 1
			self.complexity += obj_complex
			# if len(obj_dict) > 0:
			# 	self.complexity += len(obj_dict) + 1

	def accepts(self, sequence):
		if isinstance(sequence, Sequence) == False: raise TypeError("The input must be an instance of Sequence")

		if self.preloaded_sequences is not None:
			return self.preloaded_sequences[sequence.cid]

		s_list = sequence.summarize(level = "Feature")
		formula = self.query.copy()
		return seq_verify_obj(formula, s_list, subset_dicts_general)

class Conjunct():
	def __init__(self, query):
		self.query = None
		self.conjunct_type = None
		self.complexity = None

		self.preloaded_sequences = None
		self.___initialize(query)

	def ___initialize(self, query):
		if type(query) is dict:
			self.conjunct_type = "Sum"
		elif type(query) is list:
			for q_dict in query:
				if type(q_dict) is not dict:
					raise TypeError("If the query is a list, its element must be dicts")
			self.conjunct_type = "Product"
		else:
			raise TypeError("The input query must be a dict or a list of dicts")
		self.query = query

		self.complexity = 1
		if self.conjunct_type == "Product":
			for obj_dict in self.query:
				obj_complex = 0
				for spec in obj_dict.values():
					if spec != "+0" and spec != "=0": obj_complex += 1
				if obj_complex != 0: obj_complex += 1
				self.complexity += obj_complex
				# if len(obj_dict) > 0:
				# 	self.complexity += len(obj_dict) + 1
		else:
			for spec in self.query.values():
				if spec != "+0": self.complexity += 1
			# self.complexity += len(self.query)

	def preload_sequences(self, sequences):
		self.preloaded_sequences = dict({})
		for seq in sequences:
			self.preloaded_sequences.update({seq.cid: seq.satisfies(self)})
		return

	def hierarchical_rep(self, feature_order = None):
		if self.conjunct_type == "Sum":
			repr_str = "Sum Conjunct\n"
			repr_str += hierarchical_rep(self.query, base_indent = "  ", key_order = feature_order)
			return repr_str
		if self.conjunct_type == "Product":
			repr_str = "Product Conjunct\n"
			for ind, obj_dict in enumerate(self.query):
				repr_str += "  Object " + str(ind) + "\n"
				repr_str += hierarchical_rep(obj_dict, base_indent = "    ", key_order = feature_order) + "\n"
			return repr_str[:-1]

	def accepts(self, sequence):
		if isinstance(sequence, Sequence) == False: raise TypeError("The input must be an instance of Sequence")

		if self.preloaded_sequences is not None:
			return self.preloaded_sequences[sequence.cid]

		if self.conjunct_type == "Sum":
			f_dicts = sequence.summarize(level = "Feature")
			f_sum = merge_dicts(*f_dicts)
			return subset_dicts_general(self.query, f_sum)
		
		if self.conjunct_type == "Product":
			s_list = sequence.summarize(level = "Feature")
			formula = self.query.copy()
			return recur_verify_obj(formula, s_list, subset_dicts_general)

	def flattened_rep(self, feature_names, feature_lists):
		if self.conjunct_type == "Sum":
			return self.__flatten_dict(self.query,feature_names,feature_lists)
		if self.conjunct_type == "Product":
			final_rep = []
			for obj_dict in self.query: final_rep.append(self.__flatten_dict(obj_dict, feature_names, feature_lists))
			return tuple(final_rep)

	def __flatten_dict(self, query, feature_names, feature_lists):
		final_rep = []
		for f in feature_lists:
			final_rep.append([0]*len(f))
		if len(query) == 0: return tuple(final_rep)
		for key in query:
			f_pos = feature_names.index(key[0])
			fs_pos = feature_lists[f_pos].index(key[1])
			final_rep[f_pos][fs_pos] = query[key]
		return tuple(tuple(rep) for rep in final_rep)

	def __len__(self):
		return len(self.query)

	def __str__(self):
		return self.conjunct_type + " Conjunct " + self.query.__str__()

	def __repr__(self):
		return self.conjunct_type + " Conjunct " + self.query.__repr__() 

class Simple_Bayesian():
	def __init__(self, conjuncts, priors = None, llh_r = 0.1):
		self.priors = None
		self.conjuncts = None
		self.llh_r = llh_r

		self.__initialize(conjuncts, priors)

	def likelihood(self, sequence, prediction):
		likelihood_prob = np.empty(len(self.conjuncts), dtype = float)
		for ind, conj in enumerate(self.conjuncts):
			curr_pred = sequence.satisfies(conj)
			if curr_pred == prediction: likelihood_prob[ind] = 1
			else: likelihood_prob[ind] = self.llh_r
		return likelihood_prob

	def prediction(self, sequence):
		predictions = []
		for ind, conj in enumerate(self.conjuncts):
			predictions.append(sequence.satisfies(conj))
		return np.array(predictions, dtype = bool), self.priors.copy()

	def posterior(self, likelihood_prob):
		marginalizer = likelihood_prob.dot(self.priors)
		return np.multiply(likelihood_prob, self.priors)/marginalizer

	def update(self, sequence, prediction):
		self.priors = self.posterior(self.likelihood(sequence, prediction))

	def find_max(self, n = 1):
		sorted_inds = np.flip(np.argsort(self.priors))[:n]
		sorted_prob = self.priors[sorted_inds]
		sorted_conjuncts = []
		for ind in sorted_inds: sorted_conjuncts.append(self.conjuncts[ind])
		return sorted_conjuncts, sorted_prob

	def __initialize(self, conjuncts, priors):
		self.conjuncts = conjuncts
		if priors is not None:
			if len(priors) != len(conjuncts): raise RuntimeError
			self.priors = priors
		else:
			self.priors = np.ones(len(conjuncts), dtype = float)*(1/len(conjuncts))

class Disjunct():
	def __init__(self, *conjuncts):
		self.conjuncts = None
		self.conjunct_types = None
		self.complexity = None

		self.preloaded_sequences = None
		self.___initialize(*conjuncts)

	def ___initialize(self, *conjuncts):
		self.conjuncts = []
		for conj in conjuncts:
			print(conj)
			if type(conj) is not Conjunct:
				raise TypeError("The input objects must be instances of Conjunct")
			self.conjuncts.append(conj)

		self.complexity = 0
		for conj in self.conjuncts: self.complexity += conj.complexity

	def preload_sequences(self, sequences):
		self.preloaded_sequences = dict({})
		for seq in sequences:
			self.preloaded_sequences.update({seq.cid: seq.satisfies(self)})
		return

	def hierarchical_rep(self, feature_order = None):
		repr_str = "Disjunct\n"
		for conj in self.conjuncts:
			repr_str += conj.hierarchical_rep(feature_order)
			repr_str += "\nor\n"
		return repr_str[:-4]

	def accepts(self, sequence):
		if isinstance(sequence, Sequence) == False: raise TypeError("The input must be an instance of Sequence")

		if self.preloaded_sequences is not None:
			return self.preloaded_sequences[sequence.cid]

		for conj in self.conjuncts:
			if conj.accepts(sequence): return True
		return False

	def flattened_rep(self, feature_names, feature_lists):
		if self.conjunct_type == "Sum":
			return self.__flatten_dict(self.query,feature_names,feature_lists)
		if self.conjunct_type == "Product":
			final_rep = []
			for obj_dict in self.query: final_rep.append(self.__flatten_dict(obj_dict, feature_names, feature_lists))
			return tuple(final_rep)

	def __flatten_dict(self, query, feature_names, feature_lists):
		final_rep = []
		for f in feature_lists:
			final_rep.append([0]*len(f))
		if len(query) == 0: return tuple(final_rep)
		for key in query:
			f_pos = feature_names.index(key[0])
			fs_pos = feature_lists[f_pos].index(key[1])
			final_rep[f_pos][fs_pos] = query[key]
		return tuple(tuple(rep) for rep in final_rep)

	def __len__(self):
		self_len = 0
		for conj in self.conjuncts: self_len += len(conj)
		return self_len

class Simple_Bayesian():
	def __init__(self, conjuncts, priors = None, llh_r = 0.1):
		self.priors = None
		self.conjuncts = None
		self.llh_r = llh_r

		self.__initialize(conjuncts, priors)

	def likelihood(self, sequence, prediction):
		likelihood_prob = np.empty(len(self.conjuncts), dtype = float)
		for ind, conj in enumerate(self.conjuncts):
			curr_pred = sequence.satisfies(conj)
			if curr_pred == prediction: likelihood_prob[ind] = 1
			else: likelihood_prob[ind] = self.llh_r
		return likelihood_prob

	def prediction(self, sequence):
		predictions = []
		for ind, conj in enumerate(self.conjuncts):
			predictions.append(sequence.satisfies(conj))
		return np.array(predictions, dtype = bool), self.priors.copy()

	def posterior(self, likelihood_prob):
		marginalizer = likelihood_prob.dot(self.priors)
		return np.multiply(likelihood_prob, self.priors)/marginalizer

	def update(self, sequence, prediction):
		self.priors = self.posterior(self.likelihood(sequence, prediction))

	def find_max(self, n = 1):
		sorted_inds = np.flip(np.argsort(self.priors))[:n]
		sorted_prob = self.priors[sorted_inds]
		sorted_conjuncts = []
		for ind in sorted_inds: sorted_conjuncts.append(self.conjuncts[ind])
		return sorted_conjuncts, sorted_prob

	def __initialize(self, conjuncts, priors):
		self.conjuncts = conjuncts
		if priors is not None:
			if len(priors) != len(conjuncts): raise RuntimeError
			self.priors = priors
		else:
			self.priors = np.ones(len(conjuncts), dtype = float)*(1/len(conjuncts))

class Bayesian_model():
	# Prior must be in log space
	def __init__(self, conjuncts, priors = None, alpha = 0.1, beta = 0.5, gamma = 0.1, preloaded_hypothesis_space = None):
		self.priors = None
		self.conjuncts = None
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.preloaded_hypothesis_space = None

		self.__initialize(conjuncts, priors, preloaded_hypothesis_space)

	# Return log likelihood
	def likelihood(self, sequence, prediction):
		if self.preloaded_hypothesis_space is not None:
			acceptance = self.preloaded_hypothesis_space.check_accepts(sequence)
			likelihood_prob = np.zeros(len(self.conjuncts), dtype = float)
			likelihood_prob[acceptance != prediction] = np.log(self.gamma)
			return likelihood_prob

		likelihood_prob = np.empty(len(self.conjuncts), dtype = float)
		for ind, conj in enumerate(self.conjuncts):
			curr_pred = conj.accepts(sequence)
			if curr_pred == prediction: likelihood_prob[ind] = 0
			else: likelihood_prob[ind] = np.log(self.gamma)
		return likelihood_prob

	# return probability, not log probability
	def prediction(self, sequence):
		if self.preloaded_hypothesis_space is not None:
			predictions = self.preloaded_hypothesis_space.check_accepts(sequence)
		else:
			predictions = np.empty(len(self.priors))
			for ind, conj in enumerate(self.conjuncts):
				# predictions[ind] = sequence.satisfies(conj)
				predictions[ind] = conj.accepts(sequence)
		
		# Transform through alpha and beta
		predictions = np.multiply(self.alpha, predictions) + np.multiply(np.ones(len(self.priors)), (1-self.alpha)*self.beta)
		return predictions.dot(np.exp(self.priors))

	# Return log posterior
	def posterior(self, likelihood_prob):
		marginals = logsumexp(likelihood_prob + self.priors)
		return likelihood_prob + self.priors - marginals

	def update(self, sequence, prediction):
		self.priors = self.posterior(self.likelihood(sequence, prediction))

	def __initialize(self, conjuncts, priors, preloaded_hypothesis_space):
		self.conjuncts = conjuncts
		if priors is not None:
			if len(priors) != len(conjuncts): raise RuntimeError
			self.priors = priors
		else:
			# self.priors = np.ones(len(conjuncts), dtype =
			# float)*(1/len(conjuncts))
			assert False

		if preloaded_hypothesis_space is not None: self.preloaded_hypothesis_space = preloaded_hypothesis_space

class preloaded_hypothesis_space():
	def __init__(self, conjuncts, sequences):
		self.seq_to_pos = dict({})
		self.conj_mat = np.empty((len(sequences), len(conjuncts)))
		
		seq_pos = 0
		for seq in sequences:
			self.seq_to_pos.update({seq.cid: seq_pos})
			for ind, conj in enumerate(conjuncts):
				self.conj_mat[seq_pos, ind] = conj.accepts(seq)
			seq_pos += 1
	
	def check_accepts(self, seq):
		return self.conj_mat[self.seq_to_pos[seq.cid]]

# Helper Functions
###############################################################################

def hierarchical_rep(fs_dict, base_indent = "", key_order = None):
	if len(fs_dict) == 0: return base_indent + "Null"
	repr_str = ""
	all_f = []
	for key in fs_dict: all_f.append(key[0])
	unq_f = set(all_f)
	if key_order is not None:
		if not unq_f.issubset(key_order): raise KeyError
		new_f = []
		for k in key_order:
			if k in unq_f: new_f.append(k)
		unq_f = new_f
	for f in unq_f:
		repr_str += base_indent + "Feature " + str(f) + ":\n"
		for key in fs_dict: 
			if key[0] == f:
				repr_str += base_indent + "  " + str(fs_dict[key]) + " | " + str(key[1]) + "\n"
	return repr_str[:-1]

def gamma_prior(conjuncts, alpha = 1):
	complexities = []
	for conj in conjuncts: complexities.append(conj.complexity)
	prior = gamma.logpdf(complexities, alpha)
	return prior - logsumexp(prior)

def recur_verify_obj(formula, s_list, Sub_Func):
	# success: all term have been matched
	if len(formula) == 0: return True
	# failure: no term to be matched
	if len(s_list) == 0: return False

	# normal recursive operations
	match_list = []
	# for each term
	for obj_i, obj_f in enumerate(formula):
		# for each object
		for ind, obj in enumerate(s_list):
			# a term is matched
			if Sub_Func(obj_f, obj):
				# we start one branch in searching because we cannot
				# gaurentee that we should match this object with this term
				branch_formula = np.delete(formula, obj_i)
				branch_s_list = np.delete(s_list, ind)
				match_list.append(recur_verify_obj(branch_formula, branch_s_list, Sub_Func))
	# failure: no term find a matching object
	if len(match_list) == 0 or sum(match_list) == 0:
		return False
	else:
		return True

def seq_verify_obj(formula, s_list, Sub_Func):
	# for each term
	for obj_i, obj_f in enumerate(formula):
		if not Sub_Func(obj_f, s_list[obj_i]):
			return False
	return True

def merge_dicts(*dicts):
	res_dict = {}
	for d in dicts:
		for key in d:
			if key in res_dict: res_dict[key] += d[key]
			else: res_dict.update({key: d[key]})
	return res_dict

# Determine whether B is a subset of A
# Generally, B is a subset of A if B is more specific than A:
#	- Every key in A must also present in B
#   - Every keyed value in B must be equal to the keyed value in A
def subset_dicts_eq(A, B):
	if len(A) == 0: return True
	for k_a in A:
		v_a = A[k_a]
		if k_a in B:
			v_b = B[k_a]
			if v_b != v_a: return False
		else:
			return False
	return True

# Determine whether B is a subset of A
# Generally, B is a subset of A if B is more specific than A:
#	- Every key in A must also present in B
#   - Every keyed value in B must be larger or equal to the keyed value in A
def subset_dicts_geq(A, B):
	if len(A) == 0: return True
	for k_a in A:
		v_a = A[k_a]
		if k_a in B:
			v_b = B[k_a]
			if v_b < v_a: return False
		else:
			return False
	return True

def subset_dicts_general(A, B):
	if len(A) == 0: return True
	for k_a in A:
		qry_a = A[k_a]
		# if len(qry_a) != 2: raise RuntimeError("Invalid Query: ", qry_a)
		# determine the direction of the query
		drc_a = qry_a[0]
		if drc_a not in ("-", "=", "+"): raise RuntimeError("Invalid Direction in Query: ", drc_a)
		# determine the numeral of the query
		num_a = int(qry_a[1:])
		# Attribute is not present
		if k_a not in B:
			# Relation
			if k_a[1] == "Relation":
				related_feat = []
				for k_b in B.keys():
					if k_b[0] == k_a[0]:
						related_feat.append(B[k_b])
				# same rel
				if num_a == 11:
					flag = False
					for num in related_feat:
						if num == sum(related_feat): flag = True
					if flag == False: return False
					continue
				# diff rel
				elif num_a == 13:
					for num in related_feat: 
						if num > 1: return False
					continue
				else: raise RuntimeError("Invalid Query " + str(num_a))
			# Empty Numeral or Negation Case
			elif num_a == 0 or drc_a == "-":
				continue
			else:
				return False
		# Attribute is present
		ans_b = B[k_a]
		if drc_a == "-":
			if ans_b > num_a: return False
		elif drc_a == "=":
			if ans_b != num_a: return False
		else:
			if ans_b < num_a: return False
	return True

def unique_perm(elems, counts, result_list, d, constructor = tuple):
	if d < 0: yield constructor(result_list)
	else:
		for ind in range(len(elems)):
			if counts[ind] > 0:
				result_list[d] = elems[ind]
				counts[ind] -= 1
				for g in unique_perm(elems, counts, result_list, d-1, constructor = constructor):
					yield g
				counts[ind] += 1

def multiset_comb(n, r, constructor = tuple):
	"""Generate sets of size r from sample space of n elements"""
	indices = [0] * r

	yield constructor(indices)
	while True:
		# find the right-most index that does not reach the end
		for i in reversed(range(r)):
			if indices[i] != n - 1:
				break
		else:
			# if all indices are n - 1, done
			return
		# e.g. if n = 3, (0, 1, 2) --> (0, 2, 2)
		indices[i:] = [indices[i] + 1] * (r - i)
		yield constructor(indices)

def multiset_perm(n, r):
	indices = np.arange(n)
	res = [indices for i in range(r)]
	return product(*res)

def get_primes():
	""" Generate an infinite sequence of prime numbers.
	"""
	# Maps composites to primes witnessing their compositeness.
	# This is memory efficient, as the sieve is not "run forward"
	# indefinitely, but only as long as required by the current
	# number being tested.
	#
	D = {}
	
	# The running integer that's checked for primeness
	q = 2
	
	while True:
		if q not in D:
			# q is a new prime.
			# Yield it and mark its first multiple that isn't
			# already marked in previous iterations
			# 
			yield q
			D[q * q] = [q]
		else:
			# q is composite. D[q] is the list of primes that
			# divide it. Since we've reached q, we no longer
			# need it in the map, but we'll mark the next 
			# multiples of its witnesses to prepare for larger
			# numbers
			# 
			for p in D[q]:
				D.setdefault(p + q, []).append(p)
			del D[q]
		
		q += 1

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
