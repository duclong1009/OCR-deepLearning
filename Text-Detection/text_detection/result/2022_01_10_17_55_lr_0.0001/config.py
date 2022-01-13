# Input image size of craft
target_size = 768
# Input word image height of craft
height_of_box = 64.0

# Heat map size
gaussian_heatmap_size = 1024
# Region box in heat map, will be used for warping heat map to char box
start_position_region = 0.15
size_of_heatmap_region = 1 - start_position_region * 2
start_position_affinity = 0.15
size_of_heatmap_affinity = 1 - start_position_affinity * 2

# Expand char box if character splitting step obtains character box too small
expand_small_box = 5
# Expand pseudo box in watershed step for obtaining character box that have signs (signs in vietnamese)
expand_scale = 1

# Batch size
batch_size_synthtext = 1
batch_size_word = 1

epochs_end = 2000
# Decreasing lr after each 20 epoch
nb_epochs_change_lr = 20

# Path to save output
path_saved = "result"

# Path to dataset
synth_data = "E:/bkai/Dataset/vn_syn_data_craft_weak"
word_data = "E:/bkai/Dataset/Vintext"

# Path to pretrained model
pretrained_craft = "E:/bkai/Code/project/model/pretrained/craft_mlt_25k.pth"
