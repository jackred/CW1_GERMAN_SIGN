import csv
import matplotlib.pyplot as plt

def _get_file_content(filename):
    ''' loads a .csv file into the ram '''
    img = []
    with open(filename) as f:
        content = csv.reader(f, delimiter=',')
        for row in content:
            img.append(row)
    img.pop(0)
    return img

def generate_histogram(img):
    ''' generates histogram of all pixels value for all the images '''
    stat = []
    for i in range(255):
        stat.append(0)
    for row in img:
        for elem in row:
            stat[int(float(elem)) - 1] += 1
    return stat

def generate_pix_histogram(img):
    ''' generates histogram of all pixels value for each columns '''
    pix = []
    for i in range(2304):
        pix.append([])
    for j in range(255):
        for i in range(2304):
            pix[i].append(0)
    for i in range(len(img)):
        for j in range(len(img[i])):
            pix[j - 1][int(float(img[i][j])) - 1] += 1
    return pix

def generate_spec_pix_histogram(img, x):
    ''' generates histogram of all pixels value for a specific column '''
    pix = []
    for i in range(255):
        pix.append(0)
    for i in range(len(img)):
        pix[int(float(img[i][x + 1])) - 1] += 1
    return pix

def show_histogram(stat):
    ''' shows histogram '''
    print(len(stat))
    print(stat)
    plt.bar(range(1, 256),stat)
    plt.show()

def show_all_diagrams(pix):
    ''' shows all diagrams one by one '''
    for i in range(len(pixel_stat)):
        if i % 100:
            show_histogram(pixel_stat[i])

def generate_graph(filename):
    img = _get_file_content(filename)
    print('loaded')

    stat = generate_histogram(img)
    print('done basic')

    pixel_stat = generate_pix_histogram(img)
    show_histogram(pixel_stat[200])
    print('done pixel')

    spec_pix_stat = generate_spec_pix_histogram(img, 200)
    show_histogram(spec_pix_stat)

    show_all_diagrams(pixel_stat)

if __name__ == '__main__':
    generate_graph('./data/x_train_gr_smpl.csv')
