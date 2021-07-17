from pytracking.utils import TrackerParams, FeatureParams, Choice
from pytracking.features.extractor1 import MultiResolutionExtractor
from pytracking.features import deep
import torch
from pytracking.features.net_wrappers import NetWithBackbone

def parameters():

    params = TrackerParams()

    # These are usually set from outside
    params.debug = 0                         # Debug level
    params.visualization = False             # Do visualization

    # Use GPU or not (IoUNet requires this to be True)
    params.use_gpu = True

    # Feature specific parameters
    deep_params = TrackerParams()

    # Patch sampling parameters
    params.max_image_sample_size = (16 * 16) ** 2   # Maximum image sample size
    params.min_image_sample_size = (16 * 16) ** 2   # Minimum image sample size
    params.search_area_scale = 4.5                  # Scale relative to target size
    params.feature_size_odd = False                 # Good to use False for even-sized kernels and vice versa

    # Optimization parameters
    params.CG_iter = 5                    # The number of Conjugate Gradient iterations in each update after the first frame
    params.init_CG_iter = 60              # The total number of Conjugate Gradient iterations used in the first frame
    params.init_GN_iter = 6               # The number of Gauss-Newton iterations used in the first frame (only if the projection matrix is updated)
    params.post_init_CG_iter = 0          # CG iterations to run after GN
    params.fletcher_reeves = False        # Use the Fletcher-Reeves (true) or Polak-Ribiere (false) formula in the Conjugate Gradient
    params.standard_alpha = True          # Use the standard formula for computing the step length in Conjugate Gradient
    params.CG_forgetting_rate = None	  # Forgetting rate of the last conjugate direction

    # Learning parameters for each feature type
    deep_params.learning_rate = 0.0075           # Learning rate
    deep_params.output_sigma_factor = 1/4        # Standard deviation of Gaussian label relative to target size

    # Training parameters
    params.sample_memory_size = 150              # Memory size
    params.train_skipping = 10                   # How often to run training (every n-th frame)

    # Online model parameters
    deep_params.kernel_size = (4, 4)             # Kernel size of filter
    deep_params.compressed_dim = 64              # Dimension output of projection matrix
    deep_params.filter_reg = 1e-1                # Filter regularization factor
    deep_params.projection_reg = 1e-4            # Projection regularization factor

    # Windowing
    params.feature_window = False                # Perform windowing of features
    params.window_output = True                  # Perform windowing of output scores

    # Detection parameters
    params.scale_factors = torch.ones(1)        # What scales to use for localization (only one scale if IoUNet is used)
    params.score_upsample_factor = 1            # How much Fourier upsampling to use

    # Init data augmentation parameters
    params.augmentation = {'fliplr': True,
                           'rotate': [5, -5, 10, -10, 20, -20, 30, -30, 45,-45, -60, 60],
                           'blur': [(2, 0.2), (0.2, 2), (3,1), (1, 3), (2, 2)],
                           'relativeshift': [(0.25, 0.25), (-0.25, 0.25), (0.25, -0.25), (-0.25, -0.25), (0.75, 0.75), (-0.75, 0.75), (0.75, -0.75), (-0.75, -0.75)]}

    params.augmentation_expansion_factor = 2    # How much to expand sample when doing augmentation
    params.random_shift_factor = 0              # How much random shift to do on each augmented sample
    deep_params.use_augmentation = True         # Whether to use augmentation for this feature

    # Factorized convolution parameters
    # params.use_projection_matrix = True       # Use projection matrix, i.e. use the factorized convolution formulation
    params.update_projection_matrix = True      # Whether the projection matrix should be optimized or not
    params.proj_init_method = 'pca'             # Method for initializing the projection matrix  randn | pca
    params.filter_init_method = 'zeros'         # Method for initializing the spatial filter  randn | zeros
    params.projection_activation = 'none'       # Activation function after projection ('none', 'relu', 'elu' or 'mlu')
    params.response_activation = ('mlu', 0.05)  # Activation function on the output scores ('none', 'relu', 'elu' or 'mlu')

    # Advanced localization parameters
    params.advanced_localization = True         # Use this or not
    params.target_not_found_threshold = -1      # Absolute score threshold to detect target missing
    params.distractor_threshold = 100           # Relative threshold to find distractors
    params.hard_negative_threshold = 0.3        # Relative threshold to find hard negative samples
    params.target_neighborhood_scale = 2.2      # Target neighborhood to remove
    params.dispalcement_scale = 0.7             # Dispacement to consider for distractors
    params.hard_negative_learning_rate = 0.02   # Learning rate if hard negative detected
    params.hard_negative_CG_iter = 5            # Number of optimization iterations to use if hard negative detected
    params.update_scale_when_uncertain = True   # Update scale or not if distractor is close

    # Setup the feature extractor (which includes the IoUNet)
    deep_fparams = FeatureParams(feature_params=[deep_params])

    # use ResNet50 for filter
    params.use_resnet50 = True
    if params.use_resnet50:
        deep_feat_filter = deep.ATOMResNet50(output_layers=['layer3'], fparams=deep_fparams, normalize_power=2)
        params.features_filter = MultiResolutionExtractor([deep_feat_filter])

    ## IouNet parameters ##
    params.image_sample_size = 22*18
    params.search_area_scale1 = 6
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = 1.5
    params.augmentation1 = {'fliplr': True,
                           'rotate': [10, -10, 45, -45],
                           'blur': [(3,1), (1, 3), (2, 2)],
                           'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6,-0.6)],
                           'dropout': (2, 0.2)}
    params.random_shift_factor1 = 1/3
    params.box_refinement_space = 'relative'
    params.iounet_augmentation = False      # Use the augmented samples to compute the modulation vector
    params.iounet_k = 3                     # Top-k average to estimate final box
    params.num_init_random_boxes = 9        # Num extra random boxes in addition to the classifier prediction
    params.box_jitter_pos = 0.1             # How much to jitter the translation for random boxes
    params.box_jitter_sz = 0.5              # How much to jitter the scale for random boxes
    params.maximal_aspect_ratio = 6         # Limit on the aspect ratio
    params.box_refinement_iter = 10         # Number of iterations for refining the boxes
    params.box_refinement_step_length = 2.5e-3  # Gradient step length in the bounding box refinement
    params.box_refinement_step_decay = 1    # Multiplicative step length decay (1 means no decay)

    params.net = NetWithBackbone(net_path='IoUnet.pth.tar',
                                 use_gpu=params.use_gpu)

    params.vot_anno_conversion_type = 'preserve_area'

    # Segmentation parameters
    params.use_segmentation = True
    params.segm_net_path = '/home/wcz/Yang/BROST/pytracking/networks/SegmNet.pth.tar'
    params.segm_maskInitNet_path = '/home/wcz/Yang/BROST/pytracking/networks/SegmNet_maskInitNet.pth.tar'
    params.segm_use_dist = True
    params.segm_normalize_mean = [0.485, 0.456, 0.406]
    params.segm_normalize_std = [0.229, 0.224, 0.225]
    params.segm_search_area_factor = 4.0
    params.segm_surroundings_area_factor = 5.5 
    params.segm_update_skip = 10            # How often to update the background features for segmentation 
    params.segm_feature_sz = 24
    params.segm_output_sz = params.segm_feature_sz * 16
    params.segm_scale_estimation = True
    params.segm_optimize_polygon = True

    params.tracking_uncertainty_thr = 3
    params.tracking_confidence_thr = 0.3
    params.filter_sample_thr = 0.25         ## Relative threshold to filter the training samples of DCF
    params.iou_refine_thr = 0.1             ## Relative threshold to perform bounding box refine
    params.response_budget_sz = 25
    params.score_budget_sz = 10             ## Memory size for saving the confidence scores of response map
    params.uncertainty_segm_scale_thr = 3.5
    params.uncertainty_segment_thr = 10
    params.segm_pixels_ratio = 2
    params.mask_pixels_budget_sz = 25
    params.segm_min_scale = 0.2
    params.max_rel_scale_ch_thr = 0.75
    params.consider_segm_pixels_ratio = 1
    params.opt_poly_overlap_thr = 0.3
    params.poly_cost_a = 1.2
    params.poly_cost_b = 1
    params.segm_dist_map_type = 'center'        # center | bbox
    params.min_scale_change_factor = 0.95
    params.max_scale_change_factor = 1.05
    params.init_segm_mask_thr = 0.5
    params.segm_mask_thr = 0.5

    params.masks_save_path = ''
    # params.masks_save_path = 'save-masks-path'
    params.save_mask = False
    if params.masks_save_path != '':
        params.save_mask = True

    return params
