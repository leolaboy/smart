import logging
import coloredlogs
import verboselogs
import config

def get_log_fn(in_dir, out_dir):
    """
    """
    log_fn = None
    
 
    if config.params['ut'] is not None:
        # use UT specified as command line argument
        log_fn = '{}/NS.{}.log'.format(out_dir, config.params['ut'])
    else:
        # try to get UT from filenames in input directory
        fns = os.listdir(in_dir)
        for fn in fns:
            if fn.startswith('NS.'):
                log_fn = out_dir + '/' + fn[:fn.find('.', fn.find('.') + 1)] + '.log'
                break
        if log_fn is None:
            # if all else fails, use canned log file name
            log_fn = out_dir + '/nsdrp.log'
            
    if config.params['subdirs'] is False:
        # if not in "subdirs" mode than put log file in log subdirectory
        parts = log_fn.split('/')
        parts.insert(len(parts)-1, 'log')
        log_fn = '/'.join(parts)
        
    return(log_fn)


def setup_main_logger(logger, in_dir, out_dir):

    if config.params['debug']:
        logger.setLevel(logging.DEBUG)
        formatter  = logging.Formatter('%(asctime)s %(levelname)s - %(filename)s:%(lineno)s - %(message)s')
        sformatter = logging.Formatter('%(asctime)s %(levelname)s - %(filename)s:%(lineno)s - %(message)s')
    else:
        logger.setLevel(logging.INFO)
        formatter  = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
        sformatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
     
    log_fn = get_log_fn(in_dir, out_dir)
             
    if os.path.exists(log_fn):
        os.rename(log_fn, log_fn + '.prev')
         
    fh = logging.FileHandler(filename=log_fn)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
        
    if config.params['verbose'] is True:
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(sformatter)
        logger.addHandler(sh)
        
    return

if __name__=="__main__":
    # only configure logging in the main program, everywhere else just import and use it

    # set up logging
    if config.params['subdirs'] is False and config.params['cmnd_line_mode'] is False:
        logfile_folder = config.params['logdir']

   # set up obj logger
    logger.handlers = []
    
    if config.params['cmnd_line_mode'] is True:
        fn = config.params['out_dir'] + '/nsdrp.log'
    else:
        fn = config.params['out_dir'] + '/' + baseName  + '.log'

    # Check for debug flag
    if config.params['debug']:
        logger.setLevel(logging.DEBUG)
        #formatter = logging.Formatter('%(asctime)s %(levelname)s - %(filename)s:%(lineno)s - %(message)s')
        sformatter =  coloredlogs.ColoredFormatter(fmt='%(asctime)s,%(msecs)03d %(levelname)s - %(filename)s:%(lineno)s - %(message)s')
    else:
        logger.setLevel(logging.INFO)
        #formatter  = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
        sformatter = coloredlogs.ColoredFormatter(fmt= '%(asctime)s,%(msecs)03d %(levelname)s - %(message)s')


    
    # set up logging
    #logfile_folder = "N:/data/log/"
    logfile_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile_filename = f"bunker5_{logfile_date}.log"
    logfile_full = os.path.join(logfile_folder, logfile_filename)
    log_format = "%(asctime)s %(module)-15s %(levelname)-8s %(message)s"
    logging.basicConfig(
        filename=logfile_full,
        level=logging.DEBUG,
        format=log_format)
    coloredlogs.DEFAULT_LOG_FORMAT = log_format
    coloredlogs.DEFAULT_FIELD_STYLES["module"] = {"color": "magenta"}
    coloredlogs.DEFAULT_FIELD_STYLES["asctime"] = {"color": "blue", "faint": "true"}
    coloredlogs.DEFAULT_FIELD_STYLES["levelname"] = {"color": "cyan"}
    coloredlogs.DEFAULT_LEVEL_STYLES["debug"] = {"color": "black", "bright": "true"}
    coloredlogs.install(milliseconds=True)
    coloredlogs.set_level(args.loglevel)


