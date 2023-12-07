import time

# ------------------------------------------------------
# Demo functions
# ------------------------------------------------------



# region ------------------------------------------------------
# As module level functions
# ------------------------------------------------------

def atest_gen_mod( i, j=3, k=5):
    time.sleep(1)
    return {'i': i, 'j': j, 'k': k}


def atest_proc_mod(data, removek=True):
    new_dict = data.copy()  # create a copy of the original dictionary
    if removek:
        del new_dict['k']  # remove the specified key-value pair
    return new_dict


# endregion
# region ------------------------------------------------------
# As class based functions
# ------------------------------------------------------

class DemoClass:
    def atest_gen(self,i, j=3, k=5):
        time.sleep(1)
        return {'i': i, 'j': j, 'k': k}


    def atest_proc(self,data, removek=True):
        new_dict = data.copy()  # create a copy of the original dictionary
        if removek:
            del new_dict['k']  # remove the specified key-value pair
        return new_dict

# endregion