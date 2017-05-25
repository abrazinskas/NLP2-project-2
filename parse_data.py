from misc.helper import load_lexicon, load_dev_data
import pickle

lexicon = load_lexicon("data/sorted_ibm1_translations.txt", top_n=50)
data_path = "data/raw/dev1.zh-en"

print(data_path)

output_file = open('data/val/parses_max_5_top_50.pkl', 'wb')
for i, (chinese, references, Dx, Dxys) in enumerate(load_dev_data(data_path, lexicon, return_Dxy=True, max_Dxy=5)):
    print(i)
    pickle.dump([chinese, references, Dx, Dxys], output_file, pickle.HIGHEST_PROTOCOL)
output_file.close()



# output_file = open('data/test/parses_top_50.pkl', 'wb')
# for i, (chinese, references, Dx) in enumerate(load_dev_data(data_path, lexicon)):
#     print(i)
#     pickle.dump([chinese, references, Dx], output_file, pickle.HIGHEST_PROTOCOL)
# output_file.close()