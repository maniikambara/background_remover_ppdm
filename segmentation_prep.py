import os
from shutil import move

def process_dataset(root_dir, num_images=2500):
    images_dir = os.path.join(root_dir, 'images')
    masks_dir  = os.path.join(root_dir, 'masks')
    seg_dir    = os.path.join(root_dir, 'segmentation')
    
    # Validasi direktori
    for d in (images_dir, masks_dir, seg_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")
    
    # Daftar dan urutkan images berdasarkan ukuran file (desc)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
    if len(image_files) < num_images:
        raise ValueError(f"Hanya ditemukan {len(image_files)} gambar, kurang dari {num_images}.")
    
    image_sizes = [(f, os.path.getsize(os.path.join(images_dir, f))) for f in image_files]
    image_sizes.sort(key=lambda x: x[1], reverse=True)
    selected    = image_sizes[:num_images]
    selected_bases = [os.path.splitext(f)[0] for f, _ in selected]
    
    # Mapping original -> photo-N
    mapping = {orig: f'photo-{i+1}' for i, orig in enumerate(selected_bases)}
    
    # Update segmentation txt
    for split in ('train', 'val', 'trainval'):
        txt_path = os.path.join(seg_dir, f'{split}.txt')
        if not os.path.isfile(txt_path):
            continue
        with open(txt_path, 'r') as f:
            lines = [os.path.splitext(line.strip())[0] for line in f]
        new_lines = [mapping[name] for name in lines if name in mapping]
        with open(txt_path, 'w') as f:
            f.write('\n'.join(new_lines))
    
    # Rename images & masks
    for orig, new in mapping.items():
        src_img = os.path.join(images_dir, orig + '.jpg')
        dst_img = os.path.join(images_dir, new  + '.jpg')
        if os.path.isfile(src_img):
            move(src_img, dst_img)
        
        src_mask = os.path.join(masks_dir, orig + '.png')
        dst_mask = os.path.join(masks_dir, new   + '.png')
        if os.path.isfile(src_mask):
            move(src_mask, dst_mask)
    
    # Hapus file yang tidak terpilih
    for folder in (images_dir, masks_dir):
        for fname in os.listdir(folder):
            base = os.path.splitext(fname)[0]
            if base not in mapping.values():
                os.remove(os.path.join(folder, fname))

if __name__ == "__main__":
    dataset_root = r'people_segmentation'    
    process_dataset(dataset_root, num_images=2500)
    print("Dataset processing complete.")