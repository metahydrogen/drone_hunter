from bing_image_downloader import downloader

query_string = input('Enter image keyword:' )
limit = input('Enter number of images: ')

downloader.download(query_string, int(limit),  output_dir='dataset', adult_filter_off=True, force_replace=True, timeout=60)
