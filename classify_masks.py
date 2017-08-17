import PhotoScan
import glob

mask_dir = "/home/ml/felix_ws/labeling/transfer/masks/"
mask_pattern = mask_dir + "{filename}_mask.png"

doc = PhotoScan.app.document
chunk = doc.chunk
dense_cloud = doc.dense_cloud
cameras = chunk.cameras

chunk.importMasks(path=mask_pattern, method="file")
dense_cloud.selectMaskedPoints(cameras)
dense_cloud.assignClassToSelection(to=1)

## UNUSED: ##
# masklist = glob.glob(mask_pattern)
# masks_imgs = [cv2.imread(fn, 0) for fn in masklist]
# base_filenames = [fn.split('/')[-1].split('_')[0] for fn in masklist]
# cameras = [(camera.label.split('.')[0], camera) for camera in cameras]
# masks = [camera.mask for camera in cameras if camera is not None]
