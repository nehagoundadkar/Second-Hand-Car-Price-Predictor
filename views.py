from django.shortcuts import render
from .forms import carform
from django.template.loader import render_to_string
from django.http import HttpResponseRedirect
import pandas as pd
from django.shortcuts import redirect
from .models import carf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from django.core.mail import EmailMessage
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer
from django.utils import timezone
df=pd.read_csv('/home/lenovo/devtut/sales/src/carsale/sale/autos.csv',encoding='ISO-8859-1',nrows=2500)
df.drop('name',axis=1,inplace=True)
df.drop('nrOfPictures',axis=1,inplace=True)
df.drop('dateCreated',axis=1,inplace=True)
df.drop('postalCode',axis=1,inplace=True)
df.drop('lastSeen',axis=1,inplace=True)
df.drop('dateCrawled',axis=1,inplace=True)
df.drop('seller',axis=1,inplace=True)
df.drop('offerType',axis=1,inplace=True)
df.drop('abtest',axis=1,inplace=True)
df.drop('vehicleType',axis=1,inplace=True)
df.drop('gearbox',axis=1,inplace=True)
df['model'].fillna(value='unknown',inplace=True)
df['fuelType'].fillna(value='unknown',inplace=True)
df['notRepairedDamage'].fillna(value='unknown',inplace=True)
df['notRepairedDamage'].replace('ja','yes',inplace=True)
df['notRepairedDamage'].replace('nein','no',inplace=True)
#print df['brand'].unique()z
X=df[['kilometer','powerPS','yearOfRegistration','monthOfRegistration','brand','model','notRepairedDamage']]
Y=df['price']
#print df['model']
#print df.loc[df['brand']=='chevrolet','model'].unique()
#a=plt.matshow(X.corr())
#plt.colorbar(a)
ct = ColumnTransformer([("encode categorical columns", OneHotEncoder(),[4,5,6])], remainder="passthrough")
X=ct.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 40)
rfr=RandomForestRegressor()
gs=GridSearchCV(estimator=rfr,param_grid={'criterion':['mse'],'min_samples_leaf':[3],'min_samples_split':[3],'max_depth':[20],'n_estimators':[200]},cv=3)
gs=gs.fit(X_train,Y_train)
bs=gs.best_params_
regr=RandomForestRegressor(criterion=bs['criterion'],min_samples_leaf=bs['min_samples_leaf'],min_samples_split=bs['min_samples_split'],max_depth=bs['max_depth'],n_estimators=bs['n_estimators'])

model=regr.fit(X_train,Y_train)
y_true=Y_train
y_pred=regr.predict(X_train)
print (r2_score(y_true,y_pred))

def form(request):
		if request.method=='GET':
			form=carform()
			
			return render(request,'car_form.html',{'form':form})	
def predict(request):
	if request.method=='POST':
		form=carform(request.POST)
		post=form.save(commit=False)
		name=request.POST.get('name')
		km=request.POST.get('kilometers')
		power=request.POST.get('power')
		brand=request.POST.get('brand')
		model1=request.POST.get('model')
		email=request.POST.get('email')
		urd=request.POST.get('urd')
		year=request.POST.get('year')
		brand=''
		model=''
		for i in model1:
			if i=='_':
				break
			else:
				brand+=i

		mbeg=len(brand)+1
		model+=model1[mbeg:]
		post.model=model
		post.brand=brand
		post.date=timezone.now()
		post.save()
		x1=ct.transform([[km,power,year,8,brand,model,urd]])
		v1=regr.predict(x1)
		year=int(year)
		year=year-3
		x2=ct.transform([[km,power,year,8,brand,model,urd]])
		v2=regr.predict(x2)
		year=year-2
		x3=ct.transform([[km,power,year,8,brand,model,urd]])
		v3=regr.predict(x3)
		v1=int(v1)
		v2=int(v2)
		v3=int(v3)
		rendered = render_to_string('mail.html', {'name': name,'v1':v1,'brand':brand})
		email = EmailMessage('THANKS FOR CHOOSING THRIFTY WHEELS', rendered, to=[email])
		email.send()
		return render(request,'predict.html',{'v1':v1,'v2':v2,'v3':v3})
	else:
		return redirect('/form')
		
def about(request):
	#if request.method=='GET':
		#p=carf.objects.latest('date')
		return render(request,'about.html',{})
def home(request):
	if request.method=='GET':
		return render(request,'home.html',{})
def plots(request):
	return render(request,'plots.html',{})
		
# Create your views here.



