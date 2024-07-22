Pl2_scanner_ckpt_path = "/media/hdd3/neo/MODELS/2024-07-22 MegaScan DSR=8 epochs=5/8/version_0/checkpoints/epoch=4-step=8925.ckpt"
scan_mpp = 0.2297952524300848 * 8
# scan_mpp = 4
focus_region_scan_mpp_height = 64
focus_region_scan_mpp_width = 64
assert focus_region_scan_mpp_height == focus_region_scan_mpp_width, "focus_region_scan_mpp_height and focus_region_scan_mpp_width must be equal"
cropping_batch_size = 256
scanning_batch_size = 32
num_croppers = 24
num_scanners = 3
