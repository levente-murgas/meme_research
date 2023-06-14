import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import os

def plot_images_in_pdf(output_dir,json_path):
    os.makedirs(output_dir, exist_ok=True)
    # Load cluster data from JSON file
    with open(json_path, 'r') as json_file:
        cluster_data = json.load(json_file)
    for i, cluster in enumerate(cluster_data):
        print(f"{cluster['cluster_name']} ({i+1}/{len(cluster_data)})")
        plt.figure(figsize=(40 ,30))
        plt.rc('text', usetex=False)
        plt.suptitle( "Images in cluster " + cluster['cluster_name'] + " = " + str(len(cluster['images'])), fontsize=50)

        columns = 4
        # print("Fetching images from disk....")
        count=0
        count_in_page=0
        images_added = []

        # Define the PDF file using the template_label as name
        pdf_path = f"{output_dir}/{cluster['cluster_name']}.pdf"

        with PdfPages(pdf_path) as pdf:
            # Get image paths for the current cluster
            image_paths = cluster['images']
            
            # Iterate over image paths
            for path, confidence_score in image_paths.items():
                # Get filename from path
                filename = path.split('\\')[-1]
                # Load and display the image in a subplot
                img = mpimg.imread(path)
                try:
                    plt.subplot(int(12 / columns) + 1, columns, count % 12 + 1)
                    plt.axis('off')
                    plt.imshow(img)
                    plt.title(f"{filename}: {confidence_score:.4f}", fontsize=20)
                    images_added.append(path)
                    count+=1
                    count_in_page+=1

                    if count % 12 == 0:
                        pdf.savefig()
                        plt.close()
                        plt.figure(figsize=(40,30))
                        count_in_page=0
                except Exception as e:
                    print(str(e))
                    pass

                if count % 100 == 0 and count > 0:
                    plt.close()
                    break
            try: 
                if count_in_page >0:
                    pdf.savefig()
                    plt.close()  # Close the figure here
            except:
                pass

cluster_results_path = 'C:/Users/Murgi/Documents/GitHub/meme_research/outputs/clusters/jsons/embedding_results.json'
output_dir = 'C:/Users/Murgi/Documents/GitHub/meme_research/outputs/clusters/pdfs/embedding'

plot_images_in_pdf(output_dir,cluster_results_path)
