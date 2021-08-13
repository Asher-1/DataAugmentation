import os


def getFilePathList(file_dir):
    filePath_list = []
    for walk in os.walk(file_dir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list


def get_files_list(file_dir, postfix='ALL'):
    postfix = postfix.split('.')[-1]
    file_list = []
    filePath_list = getFilePathList(file_dir)
    if postfix == 'ALL':
        file_list = filePath_list
    else:
        for file in filePath_list:
            basename = os.path.basename(file)
            postfix_name = basename.split('.')[-1]
            if postfix_name == postfix:
                file_list.append(file)
    file_list.sort()
    return file_list
