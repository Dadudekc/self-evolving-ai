I'm sorry but your request contains no information about what kind of project you would like to expand on Python Flask API with advanced features including database integration, user authentication & logging etc..  But if we consider a generic use case such as creating an application that returns the current time from different routes within our web service.
   Below is one way this could be done using flask and SQLAlchemy ORM:
```python   
from datetime import datetime     # to get date-time object in python 3+ or else use timedelta instead of just 'now' for more precision time calculations..
                           
# necessary modules.  
import os                      # environment variables, required by some flask functionalities           
from sqlalchemy import SQLAlchemy     # this will be the database module we can interact with our Database via ORM(Object Relational Mapping).   
                                   # it simplifies working on databases in Python.   It is not a direct replacement for 'pymysql' or other modules like Django, Ruby etc..  But good enough     to fulfill your requirement of the project as such code will be needed only once while initializing flask application and SQLAlchemy ORM module   
from werkzeug.security import generate_password_hash   # required for password hashing in user authentication step...      
                                   
# create database url based on environment variables..  (it needs to match your DB environements)       
app = Flask(__name__)         
if os.getenv('DEVELOPMENT'):        
    app.config.from_object("configuration.DevelopmentConfig")     # loading configuration specific for current runnings environment...   For Example if we are running on local development server this will load our config file 'development'.  Or else it may default to Production Config..      
else:         
    app.config.from_object('confguration.ProductionConfig')     # loading production specific configurations for current runnings environment...   If we are not in a development server or running on the live, then this will load 'production' config file     
db = SQLAlchemy(app)                       # initializing database module..  This line is required as first thing to initialize an instance of Database.    
                                     
# User model for user authentication...        
class Users (UserMixin , db.Model):       
    id = Column(Integer, primary_key=True )         
    username  = Column(String(64), unique= True)   # Unique makes it so the email cannot be repeated      
    password_hash =  Column (String (128))        
     @property                          
     def password  (self):           return self.password_hash           
     
     def set_password(self, password: str)->None :             # Method to hash the plain text pwd before storing..    This method is called as a property for security reasons which means it will not be directly callable on instance of UserMixin class         It can only provide hashed value or change them if required.      
        self.password_hash= generate_password_hash ( password )     # hash the user entered pwd, to securely compare with stored hsh..  This is a simple way for storing and comparing encrypted version of users's Password in our database...    In Werkzeug library it has function like this.
          .checkpw(password_to_compare , self password )   # If the user entered pwd matches with stored hash, then return True otherwise False..      
        
# create routes for current time....        db is an instance of SQLAlchemy ORM and it will interact directly to database.  So we can use any model as well just specify which table in DB...     Here I am assuming there's only one route '/time'.      # @login_required(fetched=True) decorator not required if you are using UserMixin from Flask-Login package, This is added for the sake of example and wonâ€™t affect actual functionality.
@app.route('/time', methods =['GET'])    def getTime():      # Get method to fetch current time..   Using SQLAlchemy ORM instance 'db' directly interact with database using cursor object (cursor) which is easier then in python we can use â€˜â€™ query_obj= db.session .query(Model).filter(.id==1234567890)'
     return str   datetime.now()      # Return current time..    The actual SQL statement will be executed by the ORM, not directly from Python code as it is in a serverless environment and doesnâ€™t have direct access to underlying database anymore...  We are getting date-time object with help of 'datetime' module instead
```   Please note this example only demonstrates how you can set up your Flask API. You would need additional modules, endpoints (routes) based on the actual use case and authentication method required in production level code as it is sensitive information so using suitable methods to secure that data like HTTPS etc...