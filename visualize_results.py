import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from dataloader import MIMIC_CXR, CheXpertDataset
from datetime import datetime

dataset_sizes = {}
dataset_sizes['mimic'] = MIMIC_CXR.image_count
dataset_sizes['chexpert'] = CheXpertDataset.image_count

def get_results(results_folder):
    results_files_list = os.listdir(results_folder)

    results = []
    
    for i, results_file in enumerate(results_files_list):
        if results_file[-11:] == "results.txt":
            # get_results()
            results_file_reference = os.path.join(results_folder, results_file)
            with open(results_file_reference, 'r') as file:
                # Read the file line by line and store each line in a list
                lines = file.readlines()
                for j, result in enumerate(lines):
                    name_value = result.split("=")
                    metric_name = name_value[0].strip()
                    if metric_name[:11] == "All trials:":
                        if len(name_value) > 1:
                            metric_value = float(name_value[1].strip()[1:-1])
                        else:
                            metric_value = None
                        results.append([results_file, metric_value])
            
    return results

def organize_results(results):
    results_df = pd.DataFrame(results, columns=["file", "mAUC"])  
    
    results_df['cleaned_file'] = results_df['file'].apply(lambda file: file.replace("swin_base", "swin").replace("randomcnnvtran", "random_ccnvtran").replace("imagenet_21kcnnvtran", "imagenet21k_cnnvtran"))
    # convnext
    results_df['Model'] = results_df['cleaned_file'].apply(lambda file: file.split("_")[0])
    # imagenet
    results_df['Pretraining'] = results_df['cleaned_file'].apply(lambda file: file.split("_")[1])
    # 21kcnnvtran
    results_df['Project'] = results_df['cleaned_file'].apply(lambda file: file.split("_")[2])
    # mimic
    results_df['Dataset'] = results_df['cleaned_file'].apply(lambda file: file.split("_")[3])
    # ImageNet22k
    results_df['Pretraining_2'] = results_df['cleaned_file'].apply(lambda file: file.split("_")[4])
    # convnext
    results_df['Model_2'] = results_df['cleaned_file'].apply(lambda file: file.split("_")[5])
    # 25
    results_df['Annotation_Percent'] = results_df['cleaned_file'].apply(lambda file: int(file.split("_")[6]))
    # 224
    results_df['Image_Size'] = results_df['cleaned_file'].apply(lambda file: int(file.split("_")[7]))
    # exp01
    # trial01
    # results
    results_df['Experiment'] = results_df['cleaned_file'].apply(lambda file: file.split("_")[8:])  
    
    return results_df

# Create a function to format y-axis labels as percentages
def percent_formatter(x, pos):
    return f'{x:.0000%}'

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_folder", 
        dest="results_folder", 
        help="folder containing results from other runs",
        default=os.path.join("E:","Outputs","BenchmarkCNNvTransformer","MIMIC_CXR")
        )

    parser.add_argument(
        "--output_directory",
        dest="output_directory",
        help="Where to drop the images of charts that are created",
        default=os.path.join("E:","Outputs","BenchmarkCNNvTransformer","MIMIC_CXR", "charts")
        )
    
    parser.add_argument(
        "--show_plots",
        dest="show_plots",
        help="display the plots",
        action="store_true"
        )

    parser.add_argument(
        "--no_save",
        dest="no_save",
        help="don't save the plots or the summarized csv",
        action="store_true"
        )

    args = parser.parse_args()

    results_folder = args.results_folder
    output_directory = args.output_directory
    show_plots = args.show_plots
    no_save = args.no_save
    
    results = get_results(results_folder)
    results_df = organize_results(results)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    comparable_groups = results_df[["Pretraining_2", "Dataset"]].drop_duplicates()

    for _, comparable_group in comparable_groups.iterrows():
        # Assuming your DataFrame is named results_df
        # Select relevant columns
        subset_df = results_df.loc[
            (results_df["Pretraining_2"] == comparable_group["Pretraining_2"]) &
            (results_df["Dataset"] == comparable_group["Dataset"]),
            ['Model', 'Annotation_Percent', 'mAUC', 'file']
        ]

        # Group by 'model' and 'annotation_percent', calculate average of 'mAUC' and count of rows
        grouped_df = subset_df.groupby(['Model', 'Annotation_Percent']).agg({'mAUC': 'mean', 'file': 'count'})
    
        grouped_df = grouped_df.sort_values(by='Annotation_Percent', ascending=True)
    
        # Reset index to make 'model' and 'annotation_percent' columns
        grouped_df.reset_index(inplace=True)      

        grouped_df['Annotation_Percent_Label'] = grouped_df['Annotation_Percent'].apply(lambda annotation_percent: str(annotation_percent)+"%")
        grouped_df['mAUC_Label'] = grouped_df['mAUC'].apply(lambda mAUC: str((100*mAUC))+"%")

        image_count = dataset_sizes.get(comparable_group["Dataset"])

        # Iterate over unique models to create separate lines
        for model, group in grouped_df.groupby('Model'):
            plt.plot(group['Annotation_Percent'], group['mAUC'], marker='o', label=model)

            # Add data labels to points in the plot
            for i, (x, y) in enumerate(zip(group['Annotation_Percent'], group['mAUC'])):
                label = f'{str(round(group["mAUC"].iloc[i]*100, 2))}%'
                label = label + "\n"
                label = label + f"{group['file'].iloc[i]} Trials"
                dx = 0.2  # Adjust the x offset as needed
                dy = 0.001  # Adjust the y offset as needed
                #plt.text(x + dx, y + dy, label, ha='right', va='bottom', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

                plt.text(x, y + dy, label, ha='right', va='bottom', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

        def x_axis_formatter(x, pos):
            label = f'{int(round(x, 0))}%'
            label = label + "\n"
            label = label + str(int(float(image_count)*(int(x)/100))) + " images"
            return label

        # Set the formatter for the y-axis
        plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(percent_formatter))
        plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(x_axis_formatter))
        
        
        plt.grid(True)

        custom_ticks = [0, 25, 50, 75, 100]
        plt.gca().xaxis.set_major_locator(mtick.FixedLocator(custom_ticks))

        # Set labels and title
        plt.xlabel('Annotation Percent')
        plt.ylabel('mAUC')
        plt.title('mAUC vs Annotation Percent by Model')
        plt.suptitle(f'Dataset: {comparable_group["Dataset"]} | Pretraining: {comparable_group["Pretraining_2"]}')

        plt.tight_layout()

        # Display legend
        plt.legend()

        if show_plots:
            # Show the plot
            plt.show()
           
        if not no_save:
            save_file = f"{comparable_group['Dataset']}_{comparable_group['Pretraining_2']}_{timestamp}.png"
            save_file = os.path.join(output_directory, save_file)
            os.makedirs(output_directory, exist_ok=True)
            plt.savefig(save_file)
            
        plt.clf()
            
    if not no_save:
        csv_file = f"{timestamp}.csv"
        csv_file = os.path.join(output_directory, csv_file)
        results_df.to_csv(csv_file)
        
        
    
