import pandas as pd
import argparse


def main(args):
    vggRegions = pd.read_csv(args.source_csv_file)
    lblFile = pd.read_csv(args.target_csv_file)

    # initializing new column
    lblFile["Tool_and_US"] = ["" for i in lblFile.index]

    for i in vggRegions.index:
        # get the name of the current frame
        filename = vggRegions['filename'][i]

        # get the class (tool type) and shape (bounding box attributes)
        regionShape = eval(vggRegions['region_shape_attributes'][i])
        className = eval(vggRegions['region_attributes'][i])

        # as long as there is a bounding box on the frame, split attributes into dict
        if bool(className):
            tool = "ultrasound"
            bbox = {"class": className['class'], 'xmin': regionShape["x"], "ymin": regionShape["y"],
                    'xmax': regionShape["x"] + regionShape["width"], 'ymax': regionShape["y"] + regionShape["height"]}
            
            match_row_to_bbox(lblFile, filename, bbox)

            
    lblFile.to_csv(args.target_csv_file, index=False)


def match_row_to_bbox(lblFile, filename, bbox):
    """
    This function matches the bounding box just created to the corresponding row in the label csv file, without
    assuming they have the same indices
    :param lblFile: the final file that is being changed to include the new bounding box column
    :param filename: name to match the row to
    :param bbox: the bounding box attributes to be written to the label file
    """
    indexedRow = lblFile.loc[lblFile['FileName'] == filename]
    index = indexedRow.iloc[0, 0]

    lblFile.loc[index, 'Tool bounding box'].append(bbox)   

    return lblFile


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
