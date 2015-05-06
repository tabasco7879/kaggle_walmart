from multiprocessing import Process, Queue, Lock
from Queue import Empty
from walmart import EvalErr, write_submission_zero, write_submission
from walmart2 import load_data2, build_target_set4, normalize_store_data, denormalize_store_data
from models import eval_model, build_model5
import numpy as np
from similarity import sim
from sklearn import linear_model
import logging
import time

def manage_workers(workers, queue, max_worker, \
                   test_result_file, \
                   validate_only, eval_err):
    while (len(workers)>=max_worker):
        try:
            (trn, vld, tst, Y_hat2, col, cat)=queue.get_nowait()
            run_model_eval_mp(trn, vld, tst, Y_hat2, col, cat, \
                          test_result_file, \
                          validate_only, eval_err)
        except Empty:
            time.sleep(1)            
        empty_slot=-1
        for i, p in enumerate(workers):            
            if not p.is_alive():
                p.terminate()
                empty_slot=i
                break
        if empty_slot>=0:         
            del workers[empty_slot]        
            
def run_model_mp(model_fun, store_data_file, store_weather_file, test_data_file, \
               model_param=1, validate_only=False, eval_err=None, columns=None):    
    print "---------------------start here---------------------"
    worker_num=4
    test_result_file ='test_result.csv'

    with open(test_result_file, 'w') as f:
        f.write('id,units\n')
        f.close()

    store_data, store_weather, test = load_data2(store_data_file, \
          store_weather_file, test_data_file)

    store_data_max = store_data.groupby(level=1).max()

    # categorize testing data with a relevant but much smaller training set
    target_set = build_target_set4(store_data, test, store_weather, store_data_max, columns=columns)

    queue = Queue()    
    workers=[]
    for col, trn, vld, tst, cat in target_set:        
        print "item(%s), train(%d), valid(%d), test(%d), model_param(%0.2f), cat(%d)" % \
              (col, len(trn), len(vld), len(tst), model_param, cat)
        if len(tst)==0: return
        while (True):
            if len(workers)<worker_num:
                p=Process(target=model_fun, \
                        args=(queue, col, trn, vld, tst, cat, \
                              store_weather, store_data_max, \
                              model_param))
                p.start()
                workers.append(p)
                break
            manage_workers(workers, queue, worker_num, \
                           test_result_file, \
                           validate_only, eval_err)
    manage_workers(workers, queue, 1, \
                   test_result_file, \
                   validate_only, eval_err)
    
    # write out zero estimation
    if not validate_only:
        write_submission_zero(test, store_data_max, test_result_file)

    if eval_err is not None:
        e1, e2=eval_err.get_result()
        logging.info("model4(p=%f) error is: train(%f), valid(%f)" % (model_param, e1, e2))
        print "model4(p=%f) error is: train(%f), valid(%f)" % (model_param, e1, e2)

def run_model_eval_mp(trn, vld, tst, Y_hat2, col, cat, \
                  test_result_file, \
                  validate_only=False, eval_err=None):
    # evaluate error in training and validation set
    e1, e2 = eval_model(trn, vld, Y_hat2, column=col)
    print "error at item(%s.%d) is: train(%f), valid(%f)" % (col, cat, e1, e2)
    if eval_err is not None:
        eval_err.add_result(e1, len(trn), e2, len(vld))
        
    # write results to test result
    if not validate_only:
        write_submission(trn, vld, tst, Y_hat2, test_result_file, \
                         'valid_result', column=col)   

def run_model4_mp(queue, col, trn, vld, tst, cat, 
                  store_weather, store_data_max, \
                  model_param=1):    
    if cat==0:        
        Y_hat2=np.zeros((len(trn)+len(vld)+len(tst), 1))
    else:        
        nm_trn = normalize_store_data(trn, store_data_max)
        nm_vld = normalize_store_data(vld, store_data_max)
        nm_tst = normalize_store_data(tst, store_data_max)

        _,fmat = sim(nm_trn, nm_vld, nm_tst, store_weather)

        Y_hat = np.zeros((len(nm_trn) + len(nm_vld) + len(nm_tst), 1))
        X = fmat[:len(nm_trn)]

        Y = nm_trn[col].values[:,np.newaxis]
        clf = linear_model.Ridge(alpha=model_param)
        clf.fit(X, Y)
        Y_hat[:] = clf.predict(fmat)
        Y_hat2 = denormalize_store_data(trn, vld, tst, Y_hat, store_data_max, column=col)
    queue.put((trn, vld, tst, Y_hat2, col, cat))

def run_model5_mp(queue, col, trn, vld, tst, cat, 
                  store_weather, store_data_max, \
                  model_param=1):
    if cat==0:
        Y_hat2=np.zeros((len(trn)+len(vld)+len(tst), 1))
    else:
        # normalize training, validing and testing data set
        nm_trn = normalize_store_data(trn, store_data_max)
        nm_vld = normalize_store_data(vld, store_data_max)
        nm_tst = normalize_store_data(tst, store_data_max)

        Y_hat=build_model5(nm_trn, nm_vld, nm_tst, store_weather, column=col, alpha_train=model_param)

        # denormalize the sale
        Y_hat2 = denormalize_store_data(trn, vld, tst, Y_hat, store_data_max, column=col)
    queue.put((trn, vld, tst, Y_hat2, col, cat))    

def run_validation_mp(model_fun, \
                      store_data_file, store_weather_file, test_data_file, \
                      model_params=None, runs=1, validate_only=False, \
                      columns=None):
    if validate_only:        
        for p in model_params:
            eval_err_p=EvalErr()
            for i in range(runs):
                eval_err=EvalErr()
                run_model_mp(model_fun, store_data_file, \
                             store_weather_file, \
                             test_data_file, \
                             p, validate_only=True, eval_err=eval_err, \
                             columns=columns)
                eval_err_p.train_err+=eval_err.train_err
                eval_err_p.ntrain+=eval_err.ntrain
                eval_err_p.valid_err+=eval_err.valid_err
                eval_err_p.nvalid+=eval_err.nvalid
            e1, e2 = eval_err_p.get_result()
            logging.info("model(p=%f) error is: train(%f), valid(%f)" % (p, e1, e2))
    else:
        run_model_mp(model_fun, store_data_file, \
                     store_weather_file, \
                     test_data_file, \
                     model_params[0], validate_only=validate_only)
                      
if __name__ == '__main__':
    logging.basicConfig(filename='walmart.log', level=logging.INFO)    
    run_validation_mp(run_model4_mp, 
                      '../../data/store_train.txt', \
                      '../../data/store_weather.txt', \
                      '../../data/test.csv', \
                      [0.1], runs=10, validate_only=True)
