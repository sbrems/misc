def read_fits(directory,verbose=True):
    '''This routine reads all fits files data into a big data cube and all header files
    into a big header cube. The order is the same and is alphabetically.Filenames and headers
    are multiple if it was an imagecube. So all have the same length. So it returns:
    (fits)filenames,datacube,headercube'''
    #to avoid buffer overflows we need the number images first, which is not the same as
    #filenumber, as some images are cubes and some are not
    n_images = 0
    form = []
    for file in sorted(os.listdir(directory)):
        if file.endswith('.fits'):
            form = fits.getdata(directory+file).shape
            if len(form) == 3 : #image cube
                n_images += form[0]
                if verbose: print('Found ',n_images+1,' frames in ',directory)
            elif len(form) == 2: #one image
                n_images += 1
                if verbose: print('Found ',n_images,' frames in ',directory)
            else:
                raise ValueError('Fits file has unknown format!')
    
    #now make the array
    filenames = []
    headers = []
    all_data = np.full((n_images,form[-2],form[-1]),np.nan,dtype=np.float32) #float16 for memory
    n = 0
    for file in sorted(os.listdir(directory)):
        if file.endswith('.fits'):
            data,header = fits.getdata(directory+file,header=True)
            if len(data.shape) == 3 :#image cube
                all_data[n:n+data.shape[0],:,:] = data
                headers.extend(header for ii in xrange(data.shape[0]))
                filenames.extend(file for ii in xrange(data.shape[0]))
                n += data.shape[0]
            elif len(data.shape) == 2: #one image
                all_data[n,:,:] = data
                headers.append(header)
                filenames.append(file)
                
            else:
                raise ValueError('Fits file has unknown format!')

    return filenames, all_data,headers
