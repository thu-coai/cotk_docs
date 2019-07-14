Search.setIndex({docnames:["dataloader","downloader","index","metric","models/LanguageGeneration/LM-tensorflow/readme","models/LanguageGeneration/VAE-tensorflow/readme","models/LanguageGeneration/index","models/LanguageGeneration/seqGAN-tensorflow/readme","models/MultiTurnDialog/CVAE-tensorflow/readme","models/MultiTurnDialog/hred-tensorflow/readme","models/MultiTurnDialog/index","models/SingleTurnDialog/index","models/SingleTurnDialog/seq2seq-pytorch/readme","models/SingleTurnDialog/seq2seq-tensorflow/readme","notes/FAQ","notes/extend","notes/installation","notes/quickstart","notes/tutorial_cli","notes/tutorial_core","resources","wordvector"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":1,"sphinx.ext.viewcode":1,sphinx:55},filenames:["dataloader.rst","downloader.rst","index.rst","metric.rst","models/LanguageGeneration/LM-tensorflow/readme.rst","models/LanguageGeneration/VAE-tensorflow/readme.rst","models/LanguageGeneration/index.rst","models/LanguageGeneration/seqGAN-tensorflow/readme.rst","models/MultiTurnDialog/CVAE-tensorflow/readme.rst","models/MultiTurnDialog/hred-tensorflow/readme.rst","models/MultiTurnDialog/index.rst","models/SingleTurnDialog/index.rst","models/SingleTurnDialog/seq2seq-pytorch/readme.rst","models/SingleTurnDialog/seq2seq-tensorflow/readme.rst","notes/FAQ.rst","notes/extend.rst","notes/installation.md","notes/quickstart.md","notes/tutorial_cli.rst","notes/tutorial_core.rst","resources.rst","wordvector.rst"],objects:{"cotk.dataloader":{BERTLanguageProcessingBase:[0,1,1,""],BERTOpenSubtitles:[0,1,1,""],BERTSingleTurnDialog:[0,1,1,""],Dataloader:[0,1,1,""],LanguageGeneration:[0,1,1,""],LanguageProcessingBase:[0,1,1,""],MSCOCO:[0,1,1,""],MultiTurnDialog:[0,1,1,""],OpenSubtitles:[0,1,1,""],SST:[0,1,1,""],SentenceClassification:[0,1,1,""],SingleTurnDialog:[0,1,1,""],SwitchboardCorpus:[0,1,1,""],UbuntuCorpus:[0,1,1,""]},"cotk.dataloader.BERTLanguageProcessingBase":{_load_data:[0,2,1,""],all_vocab_size:[0,3,1,""],convert_bert_ids_to_ids:[0,2,1,""],convert_bert_ids_to_tokens:[0,2,1,""],convert_ids_to_bert_ids:[0,2,1,""],convert_ids_to_tokens:[0,2,1,""],convert_tokens_to_bert_ids:[0,2,1,""],convert_tokens_to_ids:[0,2,1,""],get_batch:[0,2,1,""],get_batches:[0,2,1,""],get_next_batch:[0,2,1,""],restart:[0,2,1,""],tokenize:[0,2,1,""],trim:[0,2,1,""],vocab_list:[0,3,1,""],vocab_size:[0,3,1,""]},"cotk.dataloader.BERTOpenSubtitles":{_load_data:[0,2,1,""]},"cotk.dataloader.BERTSingleTurnDialog":{_load_data:[0,2,1,""],all_vocab_size:[0,3,1,""],convert_bert_ids_to_ids:[0,2,1,""],convert_bert_ids_to_tokens:[0,2,1,""],convert_ids_to_bert_ids:[0,2,1,""],convert_ids_to_tokens:[0,2,1,""],convert_tokens_to_bert_ids:[0,2,1,""],convert_tokens_to_ids:[0,2,1,""],get_batch:[0,2,1,""],get_batches:[0,2,1,""],get_inference_metric:[0,2,1,""],get_next_batch:[0,2,1,""],get_teacher_forcing_metric:[0,2,1,""],restart:[0,2,1,""],tokenize:[0,2,1,""],trim:[0,2,1,""],vocab_list:[0,3,1,""],vocab_size:[0,3,1,""]},"cotk.dataloader.Dataloader":{get_all_subclasses:[0,4,1,""],load_class:[0,4,1,""]},"cotk.dataloader.LanguageGeneration":{_load_data:[0,2,1,""],_valid_word2id:[0,2,1,""],all_vocab_size:[0,3,1,""],convert_ids_to_tokens:[0,2,1,""],convert_tokens_to_ids:[0,2,1,""],get_batch:[0,2,1,""],get_batches:[0,2,1,""],get_inference_metric:[0,2,1,""],get_next_batch:[0,2,1,""],get_teacher_forcing_metric:[0,2,1,""],restart:[0,2,1,""],trim:[0,2,1,""],vocab_list:[0,3,1,""],vocab_size:[0,3,1,""]},"cotk.dataloader.LanguageProcessingBase":{_load_data:[0,2,1,""],_valid_word2id:[0,2,1,""],all_vocab_size:[0,3,1,""],convert_ids_to_tokens:[0,2,1,""],convert_tokens_to_ids:[0,2,1,""],get_batch:[0,2,1,""],get_batches:[0,2,1,""],get_next_batch:[0,2,1,""],restart:[0,2,1,""],trim:[0,2,1,""],vocab_list:[0,3,1,""],vocab_size:[0,3,1,""]},"cotk.dataloader.MSCOCO":{_load_data:[0,2,1,""],tokenize:[0,2,1,""]},"cotk.dataloader.MultiTurnDialog":{_load_data:[0,2,1,""],_valid_word2id:[0,2,1,""],all_vocab_size:[0,3,1,""],convert_ids_to_tokens:[0,2,1,""],convert_multi_turn_ids_to_tokens:[0,2,1,""],convert_multi_turn_tokens_to_ids:[0,2,1,""],convert_tokens_to_ids:[0,2,1,""],get_batch:[0,2,1,""],get_batches:[0,2,1,""],get_inference_metric:[0,2,1,""],get_next_batch:[0,2,1,""],get_teacher_forcing_metric:[0,2,1,""],multi_turn_trim:[0,2,1,""],restart:[0,2,1,""],trim:[0,2,1,""],vocab_list:[0,3,1,""],vocab_size:[0,3,1,""]},"cotk.dataloader.OpenSubtitles":{_load_data:[0,2,1,""],tokenize:[0,2,1,""]},"cotk.dataloader.SST":{_load_data:[0,2,1,""]},"cotk.dataloader.SentenceClassification":{_load_data:[0,2,1,""],all_vocab_size:[0,3,1,""],convert_ids_to_tokens:[0,2,1,""],convert_tokens_to_ids:[0,2,1,""],get_batch:[0,2,1,""],get_batches:[0,2,1,""],get_metric:[0,2,1,""],get_next_batch:[0,2,1,""],restart:[0,2,1,""],tokenize:[0,2,1,""],trim:[0,2,1,""],vocab_list:[0,3,1,""],vocab_size:[0,3,1,""]},"cotk.dataloader.SingleTurnDialog":{_load_data:[0,2,1,""],_valid_word2id:[0,2,1,""],all_vocab_size:[0,3,1,""],convert_ids_to_tokens:[0,2,1,""],convert_tokens_to_ids:[0,2,1,""],get_batch:[0,2,1,""],get_batches:[0,2,1,""],get_inference_metric:[0,2,1,""],get_next_batch:[0,2,1,""],get_teacher_forcing_metric:[0,2,1,""],restart:[0,2,1,""],trim:[0,2,1,""],vocab_list:[0,3,1,""],vocab_size:[0,3,1,""]},"cotk.dataloader.SwitchboardCorpus":{_load_data:[0,2,1,""],get_batch:[0,2,1,""],get_multi_ref_metric:[0,2,1,""],tokenize:[0,2,1,""]},"cotk.dataloader.UbuntuCorpus":{_load_data:[0,2,1,""],tokenize:[0,2,1,""]},"cotk.downloader":{load_file_from_url:[1,5,1,""]},"cotk.metric":{BleuCorpusMetric:[3,1,1,""],BleuPrecisionRecallMetric:[3,1,1,""],EmbSimilarityPrecisionRecallMetric:[3,1,1,""],FwBwBleuCorpusMetric:[3,1,1,""],LanguageGenerationRecorder:[3,1,1,""],MetricBase:[3,1,1,""],MetricChain:[3,1,1,""],MultiTurnBleuCorpusMetric:[3,1,1,""],MultiTurnDialogRecorder:[3,1,1,""],MultiTurnPerplexityMetric:[3,1,1,""],PerplexityMetric:[3,1,1,""],SelfBleuCorpusMetric:[3,1,1,""],SingleTurnDialogRecorder:[3,1,1,""]},"cotk.metric.BleuCorpusMetric":{close:[3,2,1,""],forward:[3,2,1,""]},"cotk.metric.BleuPrecisionRecallMetric":{_score:[3,2,1,""],close:[3,2,1,""],forward:[3,2,1,""]},"cotk.metric.EmbSimilarityPrecisionRecallMetric":{_score:[3,2,1,""],close:[3,2,1,""],forward:[3,2,1,""]},"cotk.metric.FwBwBleuCorpusMetric":{close:[3,2,1,""],forward:[3,2,1,""]},"cotk.metric.LanguageGenerationRecorder":{close:[3,2,1,""],forward:[3,2,1,""]},"cotk.metric.MetricBase":{_hash_relevant_data:[3,2,1,""],_hashvalue:[3,2,1,""],close:[3,2,1,""],forward:[3,2,1,""]},"cotk.metric.MetricChain":{add_metric:[3,2,1,""],close:[3,2,1,""],forward:[3,2,1,""]},"cotk.metric.MultiTurnBleuCorpusMetric":{close:[3,2,1,""],forward:[3,2,1,""]},"cotk.metric.MultiTurnDialogRecorder":{close:[3,2,1,""],forward:[3,2,1,""]},"cotk.metric.MultiTurnPerplexityMetric":{close:[3,2,1,""],forward:[3,2,1,""]},"cotk.metric.PerplexityMetric":{close:[3,2,1,""],forward:[3,2,1,""]},"cotk.metric.SelfBleuCorpusMetric":{close:[3,2,1,""],forward:[3,2,1,""]},"cotk.metric.SingleTurnDialogRecorder":{close:[3,2,1,""],forward:[3,2,1,""]},"cotk.wordvector":{Glove:[21,1,1,""],WordVector:[21,1,1,""]},"cotk.wordvector.Glove":{load_dict:[21,2,1,""],load_matrix:[21,2,1,""]},"cotk.wordvector.WordVector":{get_all_subclasses:[21,4,1,""],load_class:[21,4,1,""],load_dict:[21,2,1,""],load_matrix:[21,2,1,""]},cotk:{dataloader:[0,0,0,"-"],downloader:[1,0,0,"-"],metric:[3,0,0,"-"],wordvector:[21,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","classmethod","Python class method"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:classmethod","5":"py:function"},terms:{"10th":20,"125a45af618245364a722ad3fcac59534f30e64aa7e2dfefd35402cd67a74cec":19,"13386b":17,"1st":0,"2015a":20,"21it":19,"25b":17,"2nd":0,"300d":21,"3018dc317f82b6013f011c1f8ccd90c5affed710b7d7d06a7235cf455c233542":[18,19],"3rd":0,"4f101c2986f1fe10ce1d2197c3086d3659aec3e6495f381d67f00b4dbb40a538":19,"4th":0,"530d449a096671d13705e514be13c7ecffafd80deb7519aa7792950a5468549":17,"54b":19,"5c106cde":1,"68f6b8d764bff6f5440a63f87aeea97049a1d2e89942a7e524b7dabd475ffd79":15,"9e4c0afe33d98fa249e472206a39e5553d739234d0a27e055044ae8880e314b1_unzip":19,"9f1121d3988ef4789943ef18c1c0b749eec0d8eee3f12270671605ce670225f6":[18,19],"abstract":0,"break":[17,19],"case":[0,6,10,11,13,15],"class":[2,4,5,7,8,9,12,13,15,19,20,21],"default":[0,1,3,4,5,7,8,9,12,13,15,18,19,20,21],"export":9,"final":17,"float":3,"function":[0,1,3,5,15,17,18,21],"import":[0,14,17,19,20],"int":[0,3,21],"long":[3,4,5,7,8,9,12,13,19],"new":[2,12,20],"null":4,"public":2,"return":[0,1,3,15,18,19,21],"short":12,"super":[15,19],"switch":[4,5,7,8,9,12,13],"true":[0,3,4,17,19],"try":[17,19],"var":9,"while":[0,17],And:[0,4,5,9,12,13,15],But:[0,15],For:[0,4,5,7,8,9,12,13,17,18],NOT:[0,3],That:3,The:[0,1,3,4,5,7,8,9,12,13,15,17,18,19,20,21],Then:[0,15,17,19],There:[0,15,18,19],These:0,Use:[0,4,5,7,8,9,12,13,14,18],Using:[9,15],__init__:[0,15,19],__main__:14,__name__:14,_file_id:15,_file_path:15,_get_file_sha256:15,_hash_relevant_data:3,_hashvalu:3,_invalid_vocab_tim:15,_load_data:[0,15],_score:3,_test:[4,5,7],_util:[15,20],_valid_word2id:0,aaai:[7,9],abandon:0,abl:[13,18],about:[8,12,15,17,20],abov:[7,18,19],absolut:20,acceler:19,accept:[0,19],access:[9,15],accomplish:15,accord:[0,12],account:18,accuraci:[0,7],accuracymetr:0,acl:20,across:[17,19],actual:3,adam:19,adapt:[0,15],add:[2,3,4,5,7,8,9,12,13,17,19],add_metr:3,added:18,addit:[2,12],addition:[0,16],adequ:15,adjac:20,admir:7,advanc:[8,9,12,13],adversari:7,affect:0,after:[0,2,3,4,5,7,8,9,12,13,15,18,19],aggreg:20,aii:12,aim:18,air:8,aircraft:7,airplan:[5,7],airport:7,algorithm:20,align:[12,13],all:[0,3,8,15,17,18,19,20,21],all_vocab:0,all_vocab_list:[0,3],all_vocab_s:[0,3],allow:[0,3,18,20],allvocab:[0,3,15],along:8,alreadi:20,alright:8,also:[0,3,8,12,17,18,19],altern:20,alwai:[0,3,15],amazon:[15,20],amazonaw:[1,15,17,20],among:[18,20],analysi:20,ani:[0,3,15,18,19,21],annot:[0,20],annotations_trainval2017:0,anoth:[3,20],answer:3,anyon:[9,13,18],anyth:[4,5,7,8,9,12,13,18],api:19,appear:[0,21],append:[15,19],appl:19,appli:[0,12],appropri:20,area:[7,20],arg:[4,5,9,12,13,18],argument:[3,6,10,11,18],around:7,arrai:[0,3,17,19,21],art:19,artifici:7,arxiv:[3,8],ask:2,assert:[3,19],associ:18,assum:[0,3,19],atent:15,attent:[4,5,7,8,9,12,13,15],attribut:0,author:[6,10,11],autoencod:[3,8,20],automat:[0,15,17,18,21],avail:[0,3],averag:[3,7,15,20],averagelengthmetr:15,avg:[3,8],avoid:[0,3],back:[7,19],backup:9,backward:19,bad:8,bag:3,bahdanau:[12,13],ball:7,base:[0,3,21],basebal:19,baselin:[2,17],basic:[2,8,9,12,13],batch:[0,3,12,15,17,19],batch_first:19,batch_siz:[0,3,4,17,18,19],batchnorm:12,bathnorm:12,bathroom:[4,5,19],beach:7,beam:12,beamsearch:12,bear:19,beast:13,becaus:[0,15],becom:[0,12],bed:7,been:[0,1,3,8,12],befor:[0,15,18,19],begin:[0,2],behaviour:[4,5,7,8,9,13],behind:[7,19],being:[8,19],belongi:[0,20],below:[0,18,19],bench:[4,7],benchmark:[0,2],bengio:[5,9,12,13],bert:[0,16],bert_id2word:0,bert_id:0,bert_vocab:0,bert_vocab_nam:0,bertsingleturndialog:2,berttoken:0,besid:18,best:[4,5,7,8,9,12,13],better:8,between:[0,3,15,17,19],big:8,bigger:21,bike:[7,8],bit:8,black:[4,19],blanket:7,bleu:[0,3,4,7,8,9,12,13,17,18,19],bleucorpusmetr:[0,3],bleuprecisionrecallmetr:[0,3],blind:0,blue:19,board:19,boat:7,boi:[13,19],bool:[0,1,3],boot:9,both:[0,19,20],bother:0,bow:8,bowl:19,bowman:5,branch:[17,18],brick:19,broccoli:19,build:[4,9],built:18,bunch:7,bus:[5,19],buse:5,bush:19,c206893c2272af489147b80df306ee703e71d9eb178f6bb06c73cb935f474452:17,cach:[0,1,4,5,7,8,9,12,13,19,20,21],cache_dir:[1,4,5,7,8,9,12,13],cake:5,calcul:[3,15,17,19],call:[0,2,3,15,18,19,20],calle:20,caller:20,can:[0,1,3,4,5,7,8,9,12,13,15,16,17,18,19,20,21],candidate_allvocab:[0,3],candidate_allvocabs_kei:3,candidates_allvocabs_kei:3,caption:20,car:[4,5,7,8,19],castl:17,cat:[7,19],catalog:[0,20],caus:[0,4,5,7,8,9,12,13],cave:13,ceil:7,cell:[5,19],challeng:20,chang:[1,15,17],channel:20,check:[1,3,17,19],checkpoint:[4,5,7,8,9,12,13],checkpoint_max_to_keep:4,checkpoint_step:4,checksum:15,checksumdir:16,chees:7,children:5,china:[0,3],cho:[12,13],choic:12,choos:[12,15,18],christoph:21,chunk:15,citi:[5,7],class_nam:[0,21],classic:17,classif:[0,15],classmethod:[0,21],cli:2,click:[15,19],clock:[5,7,17],clone:16,close:[2,3,5,7,15,17,19],cmd:17,coai:[16,17],coco:[0,20],cocodataset:[0,20],code:[2,15,17,19],colab:19,collect:[15,17,20],color:19,com:[0,1,15,16,17,18,19,20],combin:3,come:17,command:[17,18,20],comment:9,commit:[17,18],common:[0,17,19,20],commonli:3,commun:18,compani:8,compar:[3,17],comparison:18,complet:20,complex:[3,20],compli:18,compon:3,composit:20,composition:20,comput:[3,4,5,20],condit:[0,3,8,20],confer:[5,7,12,13,20],config:[15,18],configur:18,consid:[0,3,8,17],consist:18,consortium:[0,20],constrain:20,construct:[15,17,19,20],contain:[0,2,3,15,18,19,20,21],context:[0,3,20],continu:5,contribut:15,conveni:19,convers:20,convert:[0,15,17,19],convert_bert_ids_to_id:0,convert_bert_ids_to_token:0,convert_ids_to_bert_id:0,convert_ids_to_token:[0,17,19],convert_multi_turn_ids_to_token:0,convert_multi_turn_tokens_to_id:0,convert_tokens_to_bert_id:0,convert_tokens_to_id:[0,17],copi:0,copynet:0,corpora:0,corpu:[0,20],correct:[3,19],correctli:[0,14,19],correspond:[3,15,20],cosin:3,cost:[0,4,5,7,8,9,12,13],cotk:[0,1,3,4,5,7,8,9,12,13,16,17,18,19,20,21],cotk_cach:19,cotk_record_inform:18,couch:4,count:0,countri:8,courvil:9,cover:[0,19],cow:19,cpu:[0,3,4,5,7,8,9,12,13],cpu_count:[0,3],creat:18,creator:[0,20],crossentropi:19,crossentropyloss:19,crush:13,cucumb:7,cuda:4,cur:19,current:[4,5,7,8,9,12,13,18],cut:[0,19],cvae:[2,10,20],cwd:20,dai:5,dashboard:[17,19],dashboard_result:18,data:[2,3,4,5,7,8,9,12,13,17,18,20,21],data_list:3,data_s:[0,15],dataload:[2,3,4,5,7,8,9,12,13,15,19,20],datalod:20,datapath:[4,5,7,8,9,12,13],dataset:[0,1,2,3,4,5,7,8,9,12,13,17,19],deal:15,debug:[4,5,7,8,9,12,13],decod:12,decode_mod:12,decor:5,deep:[19,20],def:[15,17,19],default_embed:21,defaultli:12,defin:[0,15,18,20],delet:1,depend:[0,19],describ:18,design:15,desk:4,detail:[0,5,7,8,9,12,13,15,17,18,20],dev:[0,4,5,7,8,9,12,13,15,17,19,20],develop:[8,17],deviat:21,dh_size:[4,12],dial:20,dialog:[0,3,8,9,12,13,20],dialogu:[0,9,20],dict:[0,3,18,19,21],did:9,differ:[0,3,15,17,18,19],digit:18,dim:19,dimens:[0,3,20,21],dine:7,dir:[4,5,7,8,9,13],directli:0,directori:[4,5,7,8,9,12,13,15,18,21],dis_accuraci:7,dis_loss:7,discard:0,discours:[3,8,20],discrimin:7,discuss:20,displai:5,distribut:[0,19,21],diverg:5,divers:[3,8,20],dl_url:17,dl_zip:17,doc:17,doe:[3,19],doesn:[9,12,14,19],doesnt:9,dog:[4,5,19],don:[0,4,5,7,8,9,12,13,15,18,19],done:8,dont:[4,5,7,8,9,12,13],doughnut:7,down:[4,5,7],download:[0,2,4,5,7,9,13,17,19,20,21],draw:19,drawn:17,drive:5,driven:20,dropout:12,droprat:12,due:0,dump:[17,18],dure:0,each:[0,3,15,19,20],easiest:9,easili:18,eat:[5,7],eccv:[0,20],edu:[0,20],edward:[0,20],effect:20,effici:3,eh_siz:12,either:12,element:[0,3],els:[9,19],embed:[0,3,12,19,21],embedding_lay:19,embedding_s:[4,19],embsimilarityprecisionrecallmetr:[0,3],emiss:8,emnlp:20,empir:20,emploi:20,empti:[0,18],encod:12,encrypt:9,end:[0,3,9,18,19],enter:[4,5,7,8,9,12,13],entri:[17,18],enumer:19,env:9,environ:[3,19],eos:[0,3,8,9,17,19],eos_id:0,epoch:[0,4,5,8,9,12,13,19],epoch_num:19,equal:[0,3,15],equip:18,error:[15,17],eskenazi:[3,8,20],especi:0,etc:[4,5,7,8,9,12,13],european:20,evalu:[0,2,3,15,17,18,20],even:19,evenli:0,everi:[0,15,18,19],everydai:20,exactli:[0,15],exampl:[0,1,3,6,10,11,13,15,17,18,20],except:[0,18],exclud:0,execut:[4,5,7,8,9,12,13],exist:[1,15,18],exit:[4,5,7,8,9,12,13,18],exmampl:3,exp:[3,12,19],experi:2,explan:0,ext_vocab:[0,15],extend:2,extern:0,extract:[0,20],extrem:17,extrema:[3,8],face:13,facilit:18,fair:[0,18],fairli:2,fals:[0,1,3,4],familiar:19,fanci:5,fast:[0,2],featur:21,februari:9,feeder:4,feet:19,fei:12,femal:20,fenc:4,fetch:17,field:[4,5,7],fighter:13,figur:9,file:[0,1,4,5,7,8,9,12,13,15,17,18,19,21],file_id:[0,15,20,21],file_path:[15,20],file_typ:0,file_util:20,filenam:[4,5,7,8,9,12,13],fill:[5,17],fin:15,find:[0,4,5,7,8,9,12,13,15,17,19,20],fine:[0,9],finish:20,fire:19,first:[0,4,5,7,8,9,12,13,15,17,19,20],fit:19,floor:7,flower:5,fly:[5,19],focu:19,folder:18,follow:[0,3,4,5,7,8,9,12,13,15,18,19,20],food:[5,7,19],forc:[0,1,8],fork:[5,7,14],form:[0,3,15,17],format:[15,19,20],former:20,fortress:13,forward:[3,15,17,19],found:17,framework:[0,19],freerun:[12,19],frequent:[2,20],frisbe:[5,19],frist:17,from:[0,1,2,3,5,7,8,12,15,17,19,20],from_pretrain:0,front:[0,4,5,19],full:[3,18],full_check:3,fulli:20,further:0,fwbw:3,fwbwbleu:3,fwbwbleucorpusmetr:[0,3],fwbwbleumetr:14,game:19,gasolin:12,gather:20,gen:[0,3,8,9,12,13,15,17,19],gen_kei:[0,3,15,17,19],gen_log_prob:[0,3,19],gen_log_prob_kei:[0,3,19],gen_loss:7,gen_prob:0,gen_prob_kei:0,gen_reward:7,gen_sentence_length:3,gener:[0,2,3,5,7,9,15,18,19,20,21],generate_sample_num:19,generated_num_per_context:[0,3],generated_token:19,get:[0,3,8,12,15,16,17,19,20],get_all_subclass:[0,21],get_batch:[0,3,15,17,19],get_inference_metr:[0,3,17,19],get_metr:[0,3,15],get_multi_ref_metr:0,get_next_batch:[0,17],get_resource_file_path:[15,20],get_teacher_forcing_metr:[0,19],gigaword5:20,giraff:5,girl:5,git:[16,17,18],github:[0,16,17,20],give:[3,15,20],given:[0,1,15,20],glass:5,global:[20,21],glove100dresourceprocessor:20,glove200dresourceprocessor:20,glove300:12,glove300d:[0,21],glove300dresourceprocessor:20,glove50d_smal:19,glove50dresourceprocessor:20,glove:[0,4,5,7,8,9,12,13,19,20,21],go_id:[0,19],going:[7,8,13],gone:9,good:7,googl:19,gpu:[4,5,7,8,9,12,13],grad_clip:4,gradient:7,graffiti:5,grass:19,great:13,greedi:12,ground:[7,8],group:[7,20],gru:[2,12],guan:7,guarante:12,guess:9,gui:12,gumbel:12,had:9,hand:5,handl:[0,20],hang:[7,12],hao:9,happen:[0,14],hard:0,hardwood:19,harm:17,has:[0,1,12,14,19,20],hash:[2,15,17],hash_sha256:15,hashlib:15,hashtag:15,hashvalu:[3,17,18,19],have:[0,1,3,4,5,7,8,9,12,13,15,16,17,18,19,20],heck:9,hello:[0,15,17],helmet:4,help:[0,4,5,7,8,9,12,13,18,19],her:[4,12,19],here:[0,3,9,13,15,17,18,19],hexdigest:15,hidden_s:19,hierarch:9,high:15,highest:13,highli:[0,3],hill:19,his:5,hold:[5,7,19],hope:15,hors:[5,17],how:[0,9,17,19],howev:[0,1,19],hred:[2,10],http:[0,1,15,16,17,18,19,20],huang:12,huh:8,hum:8,hydrant:19,hyperparamet:[4,5,7],hypothes:3,hypothesi:3,idea:9,ident:3,identif:18,ids:[0,3,17],ignor:[0,1,3,19],ignore_first_token:0,ignore_left_sampl:0,ignore_smoothing_error:3,imag:[0,13,17,20],implement:[4,5,7,9,13,15,17,19],import_local_resourc:20,inaccur:0,includ:[0,18,21],incom:19,index:[0,2,3,15,19],indic:[0,1,12,18,20,21],infer:[0,19],info:[17,19],inform:[0,12,13,18],inherit:15,init:17,initi:[0,4,5,7,8,9,12,13,17,21],input:[0,15,19],insid:0,inspect:17,instal:[2,4,5,7,8,9,12,13,19],instead:0,intellig:7,interact:20,interest:[8,20],intern:[12,13,20],introduc:[0,17,20],introduct:20,invalid:[0,3,19],invalid_vocab:[0,3,15],invalid_vocab_tim:[0,15],invok:[0,3,15],ipynb:19,isn:9,issu:14,item:[0,3],iter:[0,15,17,21],ith:21,its:[0,3,15,21],itself:0,jag:[0,3],japan:0,jeffrei:21,jian:7,john:[0,20],jointli:[12,13],joke:9,josi:12,jozefowicz:5,json:[15,17,18],jun:7,just:[0,7,8,9,15,18,19,20,21],kei:[0,3,15,19],key_nam:[0,17,19],keyboard:7,kid:12,kitchen:4,kl_loss:5,kl_weight:5,kld:5,know:[8,9,12,13,17,19],knowledg:0,known:19,konw:0,label:[0,15,20],lack:8,lai:7,lambda:15,lamp:19,languag:[0,2,3,5,6,7,20],languagegener:[2,15],languagegenerationrecord:[0,3],languagemodel:19,languageprocessingbas:[3,15],lantao:7,laptop:4,larg:[0,7,19,20],larger:12,last:[0,3,4,5,7,8,9,12,13],latest:16,latter:20,ldc97s62:[0,20],ldc:[0,20],lead:17,leader:13,learn:[0,3,5,8,12,13,19,20],least:[0,3],leav:5,left:[0,19],leg:4,len:[0,15,19,21],len_avg:15,length:[0,3,15,19,20],length_penalti:12,less:[0,3],let:17,level:[3,8,20],librari:19,like:[0,2,8,9,12,15,17,19,20],lin:[0,20],line:[4,8,9,15,17,18,19,20],linear:[19,20],linguist:[0,20],link:[15,20],lison:0,list:[0,3,15,17,18,19,21],littl:8,livelossplot:19,load:[0,2,4,5,7,8,9,12,13,15,17,21],load_class:[0,21],load_dict:21,load_file_from_url:1,load_matrix:21,load_model_from_url:1,load_pretrain_emb:21,loader:[2,17,21],local:[0,1,12,17,18,19,20,21],locat:[1,18,20],log:[0,3,4,5,7,8,9,12,13],log_dir:[4,5,7,8,9,12,13],log_softmax:3,logdir:[4,5,7,8,9,12,13],logsoftmax:19,lolol:9,longer:0,longtensor:19,look:[4,7,8,9,19],los:8,loss:[0,4,5,7,8,9,12,13,19],loss_arr:19,lot:[7,8],low:[0,20],lr_decai:4,lrec:[0,20],machin:[12,13],made:7,mai:[0,1,3,4,5,7,8,9,12,13,15,17,18,19],main:[15,17,18],maintain:15,mair:[0,20],make:[0,2,3,15,19],male:20,man:[4,5,7,20,21],manag:17,mani:19,manual:1,map:[0,3,21],mark:0,mash:7,mass:20,master:17,match:0,max:[0,12,19],max_sen_length:4,max_sent_length:[0,19],max_turn_length:[0,3],maximum:[3,20],maxin:20,mean:[0,3,4,5,7,8,9,12,13,17,19,21],mechan:[0,9,12,13],men:7,merg:15,mess:3,messag:[4,5,7,8,9,12,13,18],method:[0,9,15,20],metric:[0,2,18,19],metricbas:[3,15],metricchain:[0,3],microblog:20,microsoft:[0,20],middl:5,might:9,min:[5,19],min_kl:5,min_vocab_tim:0,minimum:20,minit:19,mirror:19,mode:[0,3,4,5,7,8,9,12,13,19],model:[0,1,2,3,6,10,11,13,15,16,20],model_config:18,model_dir:[4,5,7,8,9,12,13],model_nam:17,modul:[2,15,19],momentum:4,monitor:7,more:[0,2,3,5,7,8,13,17,18,20],most:[0,19],motorcycl:[4,5,7],mous:9,movi:[0,20],mscoco:[4,5,7,15,17,19],mscoco_dev:15,mscoco_smal:[15,17,19],mscoco_test:15,mscoco_train:15,mscocoresourceprocessor:20,mscocoresourcesprocessor:15,much:[0,8,19],multi:[0,3,20],multi_turn_context_allvocabs_kei:3,multi_turn_gen:[0,3],multi_turn_gen_kei:[0,3],multi_turn_gen_log_prob:[0,3],multi_turn_gen_log_prob_kei:[0,3],multi_turn_ref_allvocab:3,multi_turn_ref_length:3,multi_turn_reference_allvocabs_kei:3,multi_turn_reference_len_kei:3,multi_turn_trim:0,multinomi:19,multipl:[0,3,8,9,20],multiple_gen:3,multiple_gen_kei:[0,3],multiprocess:[0,3,14],multiturnbleucorpusmetr:3,multiturndialog:[2,15],multiturndialogrecord:[0,3],multiturnperplexitymetr:[0,3],must:[0,3,15],n_dim:21,name:[0,3,4,5,7,8,9,12,13,18,19,20,21],name_best:12,name_last:12,natur:[2,5,19,20],ndarrai:[0,3,19,21],ndim:21,necessari:[15,20],necessarili:18,neck:5,need:[0,3,9,18,19,20],neglect:20,neither:[0,18],net:[7,19],network:[9,12,13,19],neural:[3,8,9,12,13,19,20],neuraldialog:20,never:[0,13],new_data:15,new_nam:15,next:[0,4,8,17,19],ngram:[3,19],nice:9,night:7,nll:17,nlp:20,nlpl:[0,20],nltk:16,no_grad:19,non:[0,18],none:[0,1,3,4,5,7,8,9,12,13,17,20,21],nor:[0,18],normal:[9,12,21],northeast:[15,17,20],notcommonword:15,now:[0,8,15,19],number:[0,3,20],numpi:[0,3,16,19,21],object:[0,3,17,20],obtain:[0,20],occurr:20,ocean:[5,19],off:0,offer:0,often:0,okai:8,old:[4,7,17],onc:[3,20],one:[0,3,7,8,15,20],oneir:9,ones:[0,3,15],onli:[0,3,15,17,18,19,20],onlin:[1,15,17,18,19,20],open:[15,17,19],opensubtitl:[3,8,9,12,13],opensubtitles2016:0,opensubtitlesresourceprocessor:20,oper:[3,20],optim:19,option:[0,1,4,5,7,8,9,12,13,15,16,18,19,20],opu:[0,20],orang:4,org:[0,20],organ:3,origin:17,other:[0,2,3,4,5,7,8,9,12,13,17,18],otherwis:[0,3,21],our:[0,8,15,19],out:[0,4,7,9,13,19],out_dir:[4,5,7,8,9,12,13],outer:0,output:[3,4,5,7,8,9,12,13,15],output_lay:19,outsid:7,oven:4,over:[0,3,17,19,20,21],own:[0,15],packag:[2,6,10,11,14,17,19],pad:[0,3,15,17,19],pad_id:0,page:[4,5,7,8,9,12,13],pai:[8,15],paint:5,pair:[19,20],pan:19,paper:[5,7,8,9,12,13],parallel:0,paramet:[0,1,3,19,21],park:[4,7],pars:20,part:[12,20],pass:[0,3,18,20],passphras:9,path:[0,1,4,5,7,12,15,17,18,19,20],path_to_dump_inform:18,peak:13,pei:13,penalti:12,pennington:[20,21],pennsylvania:12,peopl:[4,5,7,8,17,19],per:[0,3],perelygin:20,perform:[0,3,6,10,11,18,20],perplex:[3,4,5,8,9,12,13,17,19],perplexity_avg_on_batch:12,perplexitymetr:[0,3,19],person:[4,5,20],philadelphia:[0,20],phone:[5,19],photo:17,php:[0,20],pickl:7,pictur:19,piec:[4,5],pineau:[9,20],pip:2,pizza:7,place:[0,1,15],plai:[5,12,19],plane:5,plate:[4,5,7],player:7,pleas:[4,5,7,17],plot:[4,5,7,8,9,12,13,19],plotloss:19,png:13,point:[8,14],polici:7,pollut:8,posit:[0,3,19],post:[0,3,8,9,12,13,19,20],post_allvocab:[0,3],post_allvocabs_kei:3,post_bert:0,post_length:0,pot:4,potato:7,pow:[0,20],pprint:19,pre:[7,19,21],precis:[3,8,9],predefin:[2,15,19,20],predict:[0,3,19],prediction_kei:0,prefix:[15,19],premium:12,prepar:2,preprint:[3,8],preprocess:[0,17,19],preprocessor:20,preset:0,pretrain:[0,2,3,4,5,7,8,9,12,13,16,19,21],pretti:[8,19],previou:[0,19],print:[3,17,19],probabl:[0,3,12],probablilti:0,problem:[4,5,7,8,9,12,13],proceed:[5,20],process:[0,3,4,5,7,8,9,12,13,15,19,20],processor:[19,20],program:14,project:[18,20],prompt:20,properti:20,protocol:18,provid:[0,2,3,17,18,19,20,21],pth:1,ptvsd:[4,5,7,8,9,12,13],publish:[2,15],pull:15,push:[17,18],put:12,python3:12,python:[2,4,5,7,8,9,12,13,16,19],pytorch:[0,1,2,3,11,16,19],qualiti:[8,15],quantiti:20,question:2,quick:[2,6,10,11],racket:5,rais:3,ran:20,random:[0,3,12],rang:19,rank:[0,20],rare:0,rate:19,rather:19,raw:20,reach:13,read:[0,8,15,19],readi:8,real:[0,8,19],realli:8,reason:8,reboot:9,recal:[3,8,19],recalcul:19,receiv:[3,17,19],recommand:15,reconstruct:[4,5],record:[3,15,20],recurs:20,red:[5,7,19],reduc:0,redund:3,ref_allvocab:3,ref_length:3,ref_sentence_length:3,refen:20,refer:[0,2,3,4,5,7,8,9,12,13,15,17,18,19,20,21],reference_allvocab:3,reference_allvocabs_kei:3,reference_len_kei:3,reference_num:0,reference_test_kei:3,regard:[0,15,20],regardless:15,registr:18,regul:15,reinstal:9,rel:[18,20],releas:[0,4,5,7,9,13,17,20],relev:3,reli:19,remot:17,remov:9,replac:0,repo:[17,18],reponam:17,report:18,repositori:[15,16],repres:[0,3],represent:[0,3,12,13,19,20,21],reproduc:2,reproduct:2,request:15,requir:[2,6,10,11,15],rerun:14,res:15,res_prefix:3,research:[0,18,20],resnet18:1,resourc:[0,2,4,12,15,17,19,21],resource_config:15,resource_processor:[15,20],resourceprocessor:20,resourcesprocessor:15,resp:[0,3,8,9,12,13],resp_allvocab:[0,3],resp_allvocabs_kei:3,resp_bert:0,resp_length:0,respons:[0,3,20],rest:21,restart:[0,17,19],restor:[4,5,7,8,9,12,13],restroom:19,result:[3,6,10,11,13,17,18,19,20],retriev:20,reward:7,rice:19,richard:21,ride:[5,7,8],right:[8,12,18,19],rkadlec:[0,20],rnn:19,road:19,robot:20,room:5,root:[18,19],row:[7,21],run:[2,4,5,7,8,9,12,13,16,17,18],run_model:[17,18],runwai:7,runxxxxxx_xxxxxx:[4,5,7,8,9,12,13],sai:8,same:[0,3,4,5,7,8,9,12,13,15,17,18],sampl:[0,3,12,15,19],samplek:12,sandwich:[5,7],satisfi:18,save:[4,5,7,9,13,18],scene:20,scope:[4,5,7,8,9,13],score:[3,18],screen:19,second:0,section:[0,18,19,20],see:[0,4,5,7,8,9,12,13,15,17,19,20],seed:[0,3],seem:2,seen:19,select:[17,20],self:[0,3,4,7,15,17,18,19],selfbleucorpusmetr:[0,3,17],selfbleumetr:14,semant:20,send:19,sent:[0,15,17,19],sent_allvocab:[0,17,19],sent_length:[0,3,17,19],sent_num:15,sentenc:[0,3,5,12,15,17,19,20],sentence_length:3,sentence_num:3,sentenceclassf:15,sentenceclassif:2,sentiment:20,separ:20,seq2seq:[2,11],seqgan:[2,6],sequenc:[0,7,12,13,18],serban:[0,9,20],serv:4,server:15,servic:20,session:[0,8],set:[0,3,4,5,7,8,9,12,13,15,17,18,19,20],sever:[3,15,17,19],sha256:15,shallow:19,shao:8,shape:[3,19,21],share:[0,17],shirt:5,shop:19,shorten:[0,19],should:[0,3,15,18,21],show:[4,5,7,8,9,12,13,18,19],show_sampl:[4,12],show_str:12,showcas:20,shower:5,shown:[4,5,7,8,9,12,13],shuffl:0,side:[5,20],sidewalk:5,sigdial:[0,20],sign:[4,5,19],similar:[0,3,15],similiar:19,simpl:19,simplest:15,simpli:16,singl:[0,3,12,13,19,20],singledialogrecord:[0,3],singleturndialog:[2,3,15],singleturndialogrecord:[0,3],sink:5,sit:[4,5,19],situat:0,size:[0,3,12,17,19,20,21],ski:[5,7],skim:17,skip:0,sky:5,slope:5,small:[0,3,4,5,8,18],smaller:[3,12],smooth:[0,3,15],snakeztc:20,snow:19,snowi:5,socher:[20,21],softmax:3,softmax_sampl:4,soldier:7,some:[0,5,7,15,17,18,19,21],somebodi:8,sometim:[0,15],somewher:[18,19],son:13,soon:[4,5,7,9,13,15],sordoni:9,sourc:[0,1,2,3,15,19,20,21],space:[5,20],speaker:20,special:[0,20],specifi:[0,3,4,5,7,8,9,12,13,15,18,21],speech:20,speed:[4,5,7,8,9,12,13,14],split:[0,17,19],splite:0,spoke:20,sst:0,sstresourceprocessor:20,stabl:16,stack:[3,19],stage:0,stand:[3,4,5,7],standard:[15,17,19,21],stanford:20,start:[0,2,3,6,10,11,14,20],state:[12,19,20],station:4,statist:20,std:[20,21],step:[0,19],stock:9,stone:17,stop:19,store:15,stove:4,str:[0,1,3,18,21],strategi:12,strawberri:7,street:[4,5,7,19],streetlight:7,string:[0,17,18,20],structur:15,studi:[6,10,11,13],stuff:8,subclass:[0,15,20,21],subject:20,subset:0,substructur:20,subtitl:[0,20],successfulli:18,suffix:18,suit:19,suitabl:19,sum:3,supervis:0,support:[0,14,18,19],suppos:18,sure:[0,3,8,19],surfboard:7,surpris:9,sutskev:[12,13],switchboard:[0,20],switchboardcorpu:8,switchboardcorpusresourceprocessor:20,system:[0,9,12,13,14,20],tabl:[4,5],take:[19,20],target:[0,21],task:[0,2,17],taxi:7,teacher:[0,12],techniqu:19,telephon:20,televis:[4,5],tell:[17,18],temporari:18,tenni:[5,7],tensor:[3,19],tensorboard:[6,10,11,13],tensorboard_plot_exampl:13,tensorboard_text_exampl:13,tensorboardx:[4,5,7,8,9,12,13],tensorflow:[2,6,10,11,16],term:0,termin:9,test:[0,3,4,5,7,8,9,12,13,15,17,19,20],text:[4,5,9,12,13,15,21],than:[0,3,8,13,19,20,21],thank:9,thats:12,thei:[0,12,14,20],them:[0,3,7,15,18,19,20],theme:12,therefor:[0,3,15,19],thi:[0,3,4,5,7,8,9,12,13,14,15,17,18,19,20],thing:[7,8],think:8,third:[0,8],thirti:7,threshold:0,through:[5,17],thu:[0,16,17],tiancheng:20,tie:4,tiedemann:[0,20],tile:19,time:[0,3,4,5,7,8,9,12,13,15,17,18,19],tip:0,tire:9,todo:3,togeth:[5,20],toi:19,toilet:[5,7,19],token:[0,3,17,18,19],token_num:15,told:12,tolist:19,too:[0,19],tool:18,top:[4,7,12],top_k:12,topic:20,topk:12,torch:[3,19],total:[0,15],tower:[5,17],town:8,tqdm:16,track:20,train:[0,2,4,5,7,8,9,12,13,15,17,20,21],train_set:15,transfer:0,translat:[12,13,20],treat:0,tree:[4,18,20],treebank:20,trim:0,trim_index:15,tsinghua:[4,20],tupl:0,turn:[0,3,8,9,12,13,19,20],turn_len_kei:3,turn_length:[0,3],tutori:[2,19],twentieth:5,twitter:20,two:[0,3,4,5,7,18,19,20],txt:[4,5,7,8,9,12,13,15,21],type:[0,3,9,15,17,18,19],ubuntu:[0,9],ubuntucorpu:[9,20],ubunturesourceprocessor:20,uncas:0,under:[15,18],understand:[0,15],unexpect:[4,5,7,8,9,13],unfair:17,unifi:17,unit:20,unk:[0,3,7,9,12,13,15,17,19],unk_id:0,unknown:[0,3,15,19],unnecessari:0,unstructur:[0,20],unsupervis:20,until:[0,20],updat:[15,19],upenn:[0,20],upload:[15,18],upstream:17,url:[1,12,15,17,18,19],usag:[2,4,5,7,8,9,12,13,15,20],use:[0,3,4,5,7,8,9,12,13,14,17,19,20],used:[0,1,3,4,5,7,8,9,12,13,17,19,20,21],useful:[2,8,9,13],user:[0,9,18],usernam:[9,17],using:[0,3,4,5,7,8,9,12,13,15,16,17,19,20],usual:[0,3,18,20],uterr:20,util:[1,2],vacabulari:0,vae:[2,6],valid:[0,3,19],valid_vocab:[0,15],valid_vocab_len:[0,15],valid_vocab_s:0,valid_vocab_tim:0,valu:[1,2,15,17],valueerror:3,vari:[0,3],variabl:[0,3,4,5,7,8,9,13],variat:[3,8,20],variou:0,vase:[5,19],vector:[2,4,5,7,8,9,12,13],vendor:7,veri:[4,5,7,8,9,12,13],version:[16,17,19,20],via:16,view:[4,5,7,8,9,12,13],vilni:5,vinyal:[5,12,13],vision:20,vocab:[0,3,19,21],vocab_list:[0,15,17,19,21],vocab_s:[0,3,17,19],vocabulari:[2,15,17,19,20],vol:9,wagon:17,wai:15,wait:[7,15],walk:[5,7,19],wang:7,want:[0,15,16,17,18,20],water:5,wave:[5,7],wear:7,websit:17,week:8,weight:[3,5,19],weinan:7,well:[7,8,18,19],were:[13,20],what:[0,8,9,12,13,15,17],whatev:0,when:[0,3,4,5,7,8,9,12,13,14,15,18,19,20],whenev:0,where:[0,1,3,8,9,17,18,19,20,21],whether:[0,1,3,17,18,19],which:[0,3,4,5,7,8,9,12,13,14,15,17,18,19,20],white:[5,7,19],who:13,whold:0,whole:[0,17],whose:19,why:[2,9],width:19,wiil:[4,5,7,8,9,13],wikipedia2014:20,window:14,winter:7,wise:3,without:[0,3,9,13,15,18],woman:[5,19],won:0,wood:4,wooden:19,word2bert_id:0,word2id:0,word2vec:[0,3],word:[0,2,3,4,5,7,8,9,12,13],word_loss:12,word_num:3,wordvec:[4,5,7,8,9,12,13,19],wordvector:[4,5,7,8,9,12,13,19,20,21],work:[0,1,8,9,12,18,20],workaround:0,working_dir:18,world:[8,15,17],would:[14,20],write:[15,17],written:[17,18],wrong:[0,1],wvclass:[4,5,7,8,9,12,13],wvpath:[4,5,7,8,9,12,13],xxxx:3,yah:9,yeah:8,yellow:[7,19],yong:7,you:[0,1,3,4,5,7,8,9,12,13,14,15,16,17,18,19,20],young:[4,5,7],your:[0,2,4,5,7,8,9,12,13,14,15,17],zebra:19,zero_grad:19,zerod:12,zhang:7,zhao:[3,8,20],zhihong:8,zhou:9,zhu:5,zip:[0,15,17,19,20],zoo:[2,17]},titles:["Data Loader","Downloader","cotk documentation","Metric","Language Model (TensorFlow)","VAE (TensorFlow)","LanguageGeneration","SeqGAN (TensorFlow)","CVAE (TensorFlow)","HRED (TensorFlow)","MultiTurnDialog","SingleTurnDialog","Seq2Seq (PyTorch)","Seq2Seq (TensorFlow)","Frequently Asked Questions","Extending Cotk: More Data, More Metrics!","Installation","Quick Start","CLI Usage: Fast Model Reproduction","GRU Language Model: Load Data and Evaluate Models","Resources","Word Vector"],titleterms:{"case":[4,5,7,8,9,12],"class":[0,3],"new":15,"public":18,For:15,Use:15,add:15,addit:19,after:14,argument:[4,5,7,8,9,12,13],ask:14,author:[5,7,8,9,12,13],auto:15,basic:[0,3],begin:14,bertlanguageprocessingbas:0,bertopensubtitl:0,bertsingleturndialog:0,call:14,cli:18,close:14,code:[14,16,18],cotk:[2,15],cvae:8,dashboard:18,data:[0,15,19],dataload:[0,17],dataset:[15,20],document:2,download:[1,15,18],evalu:19,exampl:[4,5,7,8,9,12],experi:17,extend:15,fast:18,forc:19,free:19,frequent:14,from:[14,16,18],github:18,glove100d:20,glove200d:20,glove300d:20,glove50d:20,glove50d_smal:20,gru:19,hash:[3,19],hred:9,indic:2,instal:16,languag:[4,19],languagegener:[0,6],languageprocessingbas:0,like:3,load:19,loader:0,local:15,make:18,metric:[3,14,15,17],model:[4,5,7,8,9,12,17,18,19],more:15,mscoco:[0,20],mscoco_smal:20,multiturndialog:[0,10],name:15,opensubtitl:[0,20],opensubtitles_smal:20,packag:[4,5,7,8,9,12,13],perform:[4,5,7,8,9,12,13],pip:16,predefin:17,prepar:19,publish:[17,18],pytorch:12,question:14,quick:[4,5,7,8,9,12,13,17],repositori:18,reproduc:[17,18],reproduct:18,requir:[4,5,7,8,9,12,13,16],resouc:15,resourc:20,result:[4,5,7,8,9,12],run:[14,19],seem:14,sentenceclassif:0,seq2seq:[12,13],seqgan:7,singleturndialog:[0,11],sourc:16,sst:20,start:[4,5,7,8,9,12,13,17],studi:[4,5,7,8,9,12],switchboardcorpu:[0,20],switchboardcorpus_smal:20,tabl:2,task:15,teacher:19,tensorboard:[4,5,7,8,9,12],tensorflow:[4,5,7,8,9,13],train:19,ubuntu:20,ubuntu_smal:20,ubuntucorpu:0,usag:18,use:15,vae:5,valu:[3,19],vector:[19,20,21],vocabulari:0,why:14,word:[19,20,21],your:18}})