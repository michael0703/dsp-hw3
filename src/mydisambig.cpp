#include <stdio.h>
#include "Ngram.h"
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
using namespace std;
#define INV_PROB -1e4
// --- global vars
vector<vector<string>> input_sentences;
map<string, vector<string>> ZhuYin_Lookup;
Vocab voc;
Ngram lm(voc, 2);
int max_state = -1;
int max_input = -1;
// --- end of global vars

vector<string> split_string(string s){
	istringstream s_in(s);
	string seg_term;
	vector<string> split_terms;
	while(s_in >> seg_term) {
		split_terms.push_back(seg_term);
	}
	return split_terms;
}
void read_seg_file(char* infile, char* outfile){

	ifstream f_in(infile, ios::in);
	ofstream f_out(outfile, ios::out);
	string line;
	while(getline(f_in, line)){
		
		vector<string> split_terms = split_string(line);
		input_sentences.push_back(split_terms);
		if ((int)split_terms.size() > max_input) max_input = (int)split_terms.size();
		/*
		for(int i=0; i<split_terms.size(); i++){
			printf("term size in char array: %d, %x %x\n", strlen(split_terms[i].c_str()), split_terms[i].c_str()[0],  split_terms[i].c_str()[1]);
		}*/		
	}
	f_in.close();
	f_out.close();
}
void parse_mapping(char* infile, char* outfile){
	
	ifstream f_in(infile, ios::in);
	ofstream f_out(outfile, ios::out);
	string line;
	while(getline(f_in, line)){
		vector<string> split_terms = split_string(line);
		vector<string> empty_vec;
		ZhuYin_Lookup.insert(pair<string, vector<string>>(split_terms[0], empty_vec));
		for(int i=1; i<split_terms.size(); i++){
			ZhuYin_Lookup.at(split_terms[0]).push_back(split_terms[i]);
		}
		
		if ((int)split_terms.size() - 1 > max_state){
			max_state = (int)split_terms.size() - 1;
		}
	}
}
// get index of vocab
VocabIndex get_vocIndex(string a){
	VocabIndex a_id = voc.getIndex(a.c_str());
	a_id = (a_id != Vocab_None) ? a_id : voc.getIndex(Vocab_Unknown);
	return a_id;
}

/* legacy code;
float get_prob(string a, string b = ""){
	VocabIndex context[] = {};
	if (b == ""){
		context[0] = Vocab_None;
	}
	else{
		VocabIndex b_id = voc.getIndex(b.c_str());		
		context[0] = b_id != Vocab_None ? b_id : voc.getIndex(Vocab_Unknown); 
	}
	VocabIndex a_id = voc.getIndex(a.c_str());
	a_id = (a_id != Vocab_None) ? a_id : voc.getIndex(Vocab_Unknown); 
	//cout << "this is a_id " << a_id << endl; 
	return lm.wordProb(a_id, context);
}*/

void viterbi(char* out_file){

	ofstream f_out(out_file, ios::in);
	// init viterbi for every input
	for(int input_idx = 0; input_idx < (int)input_sentences.size(); input_idx ++){
		printf("Start Num: %d sentences\n", input_idx);	
		printf("%d %d\n", max_input, max_state);
		// init first column
		float viterbi[max_input][max_state];
		int backtrack[max_input][max_state];
		VocabIndex vocabIndexLookUp[max_state][max_state];
		string observe_term = input_sentences[input_idx][0];
		vector<string> state_list = ZhuYin_Lookup.at(observe_term);
		int state_num = (int)state_list.size();
		for(int i =0; i < max_state; i ++){
			if (i >= state_num){
				break;
				//viterbi[0][i] = INV_PROB;
			}
			else{
				vocabIndexLookUp[0][i] = get_vocIndex(state_list[i]);
				VocabIndex context[] = {Vocab_None};
				viterbi[0][i] = lm.wordProb(vocabIndexLookUp[0][i], context);
				printf("observe 0, state : %d has Prob : %f\n", i, viterbi[0][i]);
			}
		}
		/* debug viterbi
		for(int i=0; i<max_state; i++){
			if (viterbi[0][i] > INV_PROB){
				printf("observe 0, state : %d has Prob : %f", i, viterbi[0][i]);
			}
		}
		induction of viterbi */
		for(int chr_idx = 1; chr_idx < (int)input_sentences[input_idx].size(); chr_idx ++ ){
			printf("working on num %d words\n", chr_idx);	
			observe_term = input_sentences[input_idx][chr_idx];			
			state_list = ZhuYin_Lookup.at(observe_term);
			state_num = (int)state_list.size();
			for(int i=0; i<max_state; i++){
				// this means O_j doesnt have state i, then this is not valid path;
				if (i >= state_num){
					break;
				}
				else{
					float maxP = -1e5;
					int maxPreIndex = -1;
					vocabIndexLookUp[chr_idx][i] = get_vocIndex(state_list[i]);
					string pre_observ = input_sentences[input_idx][chr_idx - 1];
					int PreStateLen = (int)ZhuYin_Lookup.at(pre_observ).size();	
					for(int j=0; j<max_state; j++){
						//printf("Cur StateLen: %d of %d, Pre StateLen: %d of %d\n", i, state_num, j, PreStateLen);
						if (j >= PreStateLen){
								break;
							}
							else{
								//string pre_term = ZhuYin_Lookup.at(pre_observ)[j];
								
								VocabIndex context[] = {vocabIndexLookUp[chr_idx-1][j]};
								float given_prob = lm.wordProb(vocabIndexLookUp[chr_idx][i], context);
								if (given_prob + viterbi[chr_idx - 1][j] > maxP){
									maxP = given_prob + viterbi[chr_idx-1][j];
									maxPreIndex = j;
								}
							}
					}
					viterbi[chr_idx][i] = maxP;
					backtrack[chr_idx][i] = maxPreIndex;
				}
			}
		}
		// backtracking viterbi path
		int sentence_size = (int)input_sentences[input_idx].size();
		int maxFinal = -1;
		float maxFinalP = -1e4;
		for(int st = 0; st < max_state; st++){
			if (st >= (int)ZhuYin_Lookup.at(input_sentences[input_idx][sentence_size-1]).size()) break;
			if (viterbi[sentence_size-1][st] > maxFinalP){
				maxFinalP = viterbi[sentence_size-1][st];
				maxFinal = st;
			}
		}
		vector<string> best_sentences;		
		int back_idx = sentence_size-1;
		int back_st = maxFinal;
		while(back_idx >= 0){
			string cur_term = ZhuYin_Lookup.at(input_sentences[input_idx][back_idx])[back_st];
			best_sentences.push_back(cur_term);
			if (back_idx > 0){
				back_st = backtrack[back_idx][back_st];
				back_idx -= 1;
			}
			else	break;
		}
		f_out << "<s>";	
		for(int x = sentence_size -1; x >=0; x--){
			f_out << best_sentences[x] << " ";
		}
		f_out << "</s>";
		f_out << endl;
		//if(input_idx >= 2) break;
		//break;
	}
}
int main(int argc, char** argv){
	
	char* seg_input_file = argv[1];
	char* observ_state_mapping_file = argv[2];
	char* lm_file = argv[3];
	char* out_file = argv[4]; 
	
	read_seg_file(seg_input_file, out_file);	
	printf("Max Input Length - %d\n", max_input);
	printf("Size of Input Text File : %lu\n", input_sentences.size());
	parse_mapping(observ_state_mapping_file, out_file);
	printf("Max State of ZhuYin - mapping - %d\n", max_state);
	printf("Map size of ZhuYin Mapping File : %lu\n", ZhuYin_Lookup.size());
	
	File _lm(lm_file, "r");
	lm.read(_lm);
	_lm.close();
	
	
	viterbi(out_file);	
	/* example code only
	VocabIndex vid = voc.getIndex(input_sentences[0][0].c_str());
	printf("%d %d\n", input_sentences[0][0].c_str()[0], input_sentences[0][0].c_str()[1]);
	cout << "example vid : " << vid << endl;
	VocabIndex context[] = {Vocab_None};
	printf("log Prob = %f\n", lm.wordProb(vid, context));
	*/
	return 0;	
}
