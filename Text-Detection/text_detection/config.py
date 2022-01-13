# Input image size of word_image_out_from_craft
target_size = 768
# Input word image height of word_image_out_from_craft
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
batch_size_synthtext = 2
batch_size_word = 5

epochs_end = 2000
# Decreasing lr after each 20 epoch
nb_epochs_change_lr = 5

# Path to save output
path_test_folder = "test_folder/image"
path_saved_train = "model/craft"
path_saved_craft_inference = "test_folder/result/craft"
path_saved_linkrefiner_inference = "test_folder/result/linkrefiner"

# Path to dataset
synth_data = "/content/drive/MyDrive/BK_AI/Dataset/vn_syn_data_craft_weak"
word_data = "/content/drive/MyDrive/BK_AI/Dataset/VinText"

# Path to pretrained model

pretrained_craft = "model/craft/2022_01_12_06_13_lr_0.0001/craft_mlt_25k.pth"
pretrained_linkrefiner = "model/pretrained/craft_refiner_CTW1500.pth"
