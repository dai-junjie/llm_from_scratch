import time
def llm_api():
    
    num  = 14
    batch_size = 3
    
    item = []
    
    for i in range(num):
        item.append(i)
        if len(item) == batch_size:
            yield item
            item = []
            time.sleep(1)
        
    if item:
        yield item
        
for chunk in llm_api():
    print(chunk)