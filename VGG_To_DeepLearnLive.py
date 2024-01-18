import pandas as pd
import argparse


def main(args):
    # look into parsers so i don't have to manually change this every time
    vggRegions = pd.read_csv(args.source_csv_file)
    lblFile = pd.read_csv(args.target_csv_file)

    # initializing new columns
    lblFile["Tool_and_US"] = ["" for i in lblFile.index]
    lblFile['Tool bounding box'] = [[] for j in lblFile.index]

    for i in vggRegions.index:
        # are the vgg and label file indices the same?
        filename = vggRegions['filename'][i]

        regionShape = eval(vggRegions['region_shape_attributes'][i])
        className = eval(vggRegions['region_attributes'][i])
        if bool(className):
            tool = "ultrasound"
            bbox = {"class": className['class'], 'xmin': regionShape["x"], "ymin": regionShape["y"],
                    'xmax': regionShape["x"] + regionShape["width"], 'ymax': regionShape["y"] + regionShape["height"]}

            indexedRow = lblFile.loc[lblFile['FileName'] == filename]
            index = indexedRow.iloc[0, 0]

            lblFile.loc[index, 'Tool bounding box'].append(bbox)
            lblFile.loc[index, 'Tool_and_US'] = tool

    lblFile.to_csv(args.target_csv_file, index=False)


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    # Data
    parser.add_argument(
        '--source_csv_file',
        type=str,
        default='',
        help='Csv file from VGG annotations'
    )
    parser.add_argument(
        '--target_csv_file',
        type=str,
        default='',
        help='Csv file compatible with deep learn live'
    )

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", parents=[get_arguments()], add_help=False)
    args = parser.parse_args()

    main(args)
