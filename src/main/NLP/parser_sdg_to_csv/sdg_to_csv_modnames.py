import pymongo
import pymysql
import csv
#from main.CONFIG_READER.read import get_details

class SDG_CSV_RESULTS():
    
    def __init__(self):
            self.client = '127.0.0.1'
            self.database = '123'
            self.port = 3306
            self.username = 'yzyucl'
            self.password = 'Yzy8588903'
            self.driver = '{MySQL ODBC 8.0 Unicode Driver}'
        # self.database = get_details("SQL_SERVER", "database")
        # self.username = get_details("SQL_SERVER", "username")
        # self.client = get_details("SQL_SERVER", "client")
        # self.password = get_details("SQL_SERVER", "password")
        # self.port = 3306
    
    def generate_csv_file(self,sdg_goal):
        con_mongo = pymongo.MongoClient('localhost', 27017)
        con_sql = pymysql.connect(host=self.client, port=self.port, db=self.database, user=self.username, password=self.password)
        cursor = con_sql.cursor(pymysql.cursors.DictCursor)
        db = con_mongo.miemie
        collection = db.MatchedModules
        Faculty = ['Faculty of Arts and Humanities','Faculty of Social and Historical Sciences','Faculty of Brain Sciences','Faculty of Life Sciences','Faculty of the Built Environment', 'School of Slavonic and Eastern European Studies'
                   ,'Institute of Education', 'Faculty of Engineering Science',' Faculty of Maths & Physical Sciences', 'Faculty of Medical Sciences','Faculty of Pop Health Sciences',' Faculty of Laws']
        sdg_file1 = sdg_goal
        sdg_file = sdg_file1.replace(" ",'')
        if len(sdg_file) == 8:
            sdg_file = sdg_file[2:6]
        elif len(sdg_file) == 9 and sdg_file[6] != "\"":
            sdg_file = sdg_file[2:7]
        else:
            sdg_file = sdg_file[2:6]
        sdg1_list_id = []
        result = collection.find({"Related_SDG": {'$regex': sdg_goal}})
        for i in result:
            sdg1_list_id.append(i["Module_ID"])
        sdg1_list_faculty = []
        sql = "SELECT * FROM moduledata"
         # execute SQL
        cursor.execute(sql)
        # get SQL data
        results = cursor.fetchall()
        for row in results:
            id = row['Module_ID']
            faculty = row['Faculty']
            for i in sdg1_list_id:
                if i == row['Module_ID']:
                    sdg1_list_faculty.append(faculty)                
        # close SQL
        con_sql.close()
        
        
        with open("parser_sdg_to_csv/"+ sdg_file + ".csv","w+",encoding='utf-8') as file:
             csv_writer = csv.writer(file)
             csv_writer.writerow(["Faculty","Number"])
             for i in range(0,len(Faculty)):
                 csv_writer.writerow([Faculty[i],sdg1_list_faculty.count(Faculty[i])])

        #added this below to get a csv file that is grouped by [Faculty Name,...all the modules corresponding to that sdg...]
        with open("parser_sdg_to_csv/"+ sdg_file + "_modnames.csv","w+",encoding='utf-8') as file:
             csv_writer = csv.writer(file)
            #  csv_writer.writerow(["Faculty","Number"])
             for i in range(0,len(Faculty)):
                 modnames_list = []
                 for j in range( 0, len(sdg1_list_faculty)):
                     if sdg1_list_faculty[j] == Faculty[i]:
                         modnames_list.append(sdg1_list_id)
                 csv_writer.writerow([Faculty[i], modnames_list])

        
                 
    def run(self):
        sdg_goals = ['.*SDG 1".*','.*SDG 2.*','.*SDG 3.*','.*SDG 4.*','.*SDG 5.*','.*SDG 6.*','.*SDG 7.*','.*SDG 8.*','.*SDG 9.*','.*SDG 10.*','.*SDG 11.*','.*SDG 12.*','.*SDG 13.*','.*SDG 14.*','.*SDG 15.*','.*SDG 16.*','.*SDG 17.*']  
        for i in sdg_goals:
            self.generate_csv_file(i)

SDG_CSV_RESULTS().run()