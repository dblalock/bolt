import subprocess

# so the way this works is:
#   -data is hosted on yandex disk (like google drive)
#   -you have to curl the user-facing url for the file to get the download link
#   -given the download link, you can wget the file

yadiskLink = "https://yadi.sk/d/11eDCm7Dsn9GA"
BASE_URL = '"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={}&path='.format(yadiskLink)
CURL = '/usr/bin/curl '  # note the trailing space


def download_path(path):
        command = CURL + BASE_URL + path + '"'

        print "-> curl command: ", command

        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        (out, err) = process.communicate()
        wgetLink = out.split(',')[0][8:]
        wgetCommand = 'wget ' + wgetLink + ' -O ' + path.strip('/')
        print "Downloading file: " + path + ' ...'

        print wgetCommand

        # process = subprocess.Popen(wgetCommand, stdin=subprocess.PIPE, shell=True)
        # process.stdin.write('e')
        # process.wait()


def download_base():  # download base files
    # for i in xrange(37):
    for i in xrange(2):
        path = '/base/base_' + str(i).zfill(2)
        download_path(path)


def download_learn():      # download learn files
    # for i in xrange(14):
    for i in xrange(2):
        path = '/learn/learn_' + str(i).zfill(2)
        download_path(path)

        # command = CURL + BASE_URL + '/learn/learn_' + str(i).zfill(2) + '"'

        # print "-> curl command: ", command

        # process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        # (out, err) = process.communicate()
        # wgetLink = out.split(',')[0][8:]
        # wgetCommand = 'wget ' + wgetLink + ' -O learn_' + str(i).zfill(2)
        # print "Downloading learn chunk " + str(i).zfill(2) + ' ...'
        # print wgetCommand
        # process = subprocess.Popen(wgetCommand, stdin=subprocess.PIPE, shell=True)
        # process.stdin.write('e')
        # process.wait()


def main():
    # download_base()
    # download_learn()
    download_path('/deep10M.fvecs')
    download_path('/deep1B_groundtruth.ivecs')
    download_path('/deep1B_queries.fvecs')


if __name__ == '__main__':
    main()
