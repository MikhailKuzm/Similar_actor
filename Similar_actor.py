import os  
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image
from io import BytesIO
import time 


driver = webdriver.Firefox()
driver.get("https://www.kino-teatr.ru/kino/acter/all/ros/a/")

for letter in range(11, 27):
    #переходим на  следующую страницу, за исключением первой итерации
    if letter != 1:  
        xpath = f'//*[@id="all_body_block"]/div[4]/div/div[3]/div[1]/div[3]/div[1]/a[{letter}]'
        next_page = driver.find_element(By.XPATH,  xpath)
        next_page.click()

    time.sleep(3)
    #находим количество актёров на одной странице
    act_per_page = driver.find_elements_by_class_name('list_item_name')
    for i in range(3, len(act_per_page)+1)[::5]:
        #переходим на страницу актёра
        actor_page = driver.find_element_by_xpath(f'//*[@id="all_body_block"]/div[4]/div/div[3]/div[1]/div[3]/div[{i}]/div[2]/div[1]/h4/a/strong')
        actor_page.click()

        time.sleep(2)
        #определяем пол: надпись актёр/актриса
        try:
            gender = driver.find_element_by_xpath('//*[@id="actor_table_block"]/div[4]')
            gender = 'men' if gender.text == 'Актёр' else 'women'
        except Exception:
            driver.back()
            time.sleep(2)
            continue

        #определяем имя  
        name = driver.find_element(By.CLASS_NAME, 'actor_header')
        act_name = name.text.split('\n')[0]
        act_name = act_name.replace(' ', '_')
        print(act_name)
        
        #определяем количество фото в альбоме, если фото нет, переходим к следующему актёру
        try:
            photo = driver.find_elements(By.PARTIAL_LINK_TEXT,  'фото')
            if len(photo) > 1:
                num_pict = photo[0].get_attribute('text')
                num_pict = int(num_pict.split(' ')[0])
                print(num_pict)
            else:
                print("no pictures")
                continue
        except Exception:
            driver.back()
            time.sleep(2)
            continue
    
        #переходим в альбом
        albom = driver.find_element_by_partial_link_text('Перейти в фотоальбом')
        albom.click()
        print('перешли в альбом')
        
        time.sleep(2)
        #создаём папку с фото актёра
        os.mkdir(f'.\\dataset\\{gender}\\{act_name}')

        #скачиваем фотографии
        classes = driver.find_elements_by_class_name('prev_block_pic')
        if num_pict>12:
            for t in range(12): 
                image_url = classes[t].get_attribute('src') 
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                img.save(f'.\\dataset\\{gender}\\{act_name}\\{t}.jpeg')
        else:
            for t in range(num_pict): 
                image_url = classes[t].get_attribute('src') 
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                img.save(f'.\\dataset\\{gender}\\{act_name}\\{t}.jpeg')

        
        driver.back()
        time.sleep(2)
        driver.back()

driver.quit()
