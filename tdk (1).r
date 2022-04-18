
install.packages("readxl")
getwd()

library(readxl) #Adatok beolvasásához
library(lmtest) #Wald-tesztekhez
library(ggplot2) #Ábrázoláshoz
library(psych) #Leíró statisztikához
library(StatMeasures) #Konfúziós mátrixhoz
library(MASS) #Akaike információs kritériumhoz
library(pROC) #ROC görbe rajzoláshoz
library(randomForest) #Véletlen erdő modellhez
library(DescTools) #Pseudo-R^2-hez

#### Logisztikus regresszió ####

# Adatok beolvasása és az adatok felosztása:

heart = read_excel("heartmasolata1.xlsx")

lapply(heart[,1:21], table) #arányok

heart[,-c(5,16,15)] = lapply(heart[,c(-5,-15,-16)], as.factor) #kategóriális változók alakítása

n = nrow(heart)

training = sample(1:n, size = round(0.9*n), replace = F)
train = heart[training,]  #Tanuló és teszt adatsorra bontás
test = heart[-training,]



# A kipróbált modellek:

model1=glm(HeartDiseaseorAttack~.-Education, train, family = binomial(link="logit"))
summary(model1)

model2=glm(HeartDiseaseorAttack~.-Education-BMI+log(BMI), train, family = binomial(link="logit"))
summary(model2)

model3=glm(HeartDiseaseorAttack~.-Education-BMI+log(BMI)-PhysHlthdum-AnyHealthcare, train, family = binomial(link="logit"))
summary(model3)

model4=glm(HeartDiseaseorAttack~.-Education-BMI+log(BMI)-PhysHlthdum-AnyHealthcare-Green, train, family = binomial(link="logit"))
summary(model4)

#modellszelekció:

anova(model2,model3)
waldtest(model2,model4,test="F")

#R^2 és a modellek összehasonlítása

PseudoR2(model2,"McFadden")
PseudoR2(model3,"McFadden")
PseudoR2(model4,"McFadden")

# LEGJOBB MODELL 

model_vegleges = glm(HeartDiseaseorAttack~. -Education -BMI + log(BMI) -AnyHealthcare, train, family = binomial(link = "logit"))

summary(model_vegleges)

PseudoR2(model_vegleges, "McFadden")

#### ROC-görbe és klasszifikáció: ####

table(heart$HeartDiseaseorAttack)

#klasszifikáció:

test$predicted = predict(model_vegleges, test, type = "response")

ROC = roc(train$HeartDiseaseorAttack, model_vegleges$fitted.values)

roc = ggroc(ROC, legacy.axes = T, col = "red")

auc(ROC)

#legjobb cut-value:

bestcutoff = coords(ROC, "best", ret = "threshold", best.method = "youden")

test$becsult = test$predicted > bestcutoff$threshold

#0,5-ös Cut-value:

test$becsult2 = test$predicted > 0.5 

#Konfúziós mátrixok:

table(test$HeartDiseaseorAttack, test$becsult)
table(test$HeartDiseaseorAttack, test$becsult2) 

# ROC-GÖRBE ÁBRÁZOLÁSA:

roc + xlab("1 - Specificitás")+ylab("Szenzitivitás")+
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color = "green", linetype = "dashed")+
  theme_minimal()+
  ggtitle("A végleges logisztikus modell ROC-görbéje")+
  theme(text = element_text(family = "serif"),
        plot.subtitle = element_text(hjust = 0.5, size = 15),
        plot.title = element_text(hjust = 0.5, size = 19),            
        axis.title = element_text(size = 15),
        axis.text = element_text(size = 13),
        panel.border = element_rect(fill = NA, colour = "black", size = 1))



#### Véletlen erdő modell ####

#véletlen erdő eredeti arányokkal

random = randomForest(HeartDiseaseorAttack~., train, mtry = 5 , ntree = 128)

summary(random)

random

test$predicted2 = predict(random, test)

table(test$HeartDiseaseorAttack, test$predicted2)


#### Random forest megváltoztatott arányokkal ####

#betegek:

heart = read_excel("heartmasolata1.xlsx")

beteg = train[heart$HeartDiseaseorAttack > 0,]

eg = train[heart$HeartDiseaseorAttack < 1,]

egeszseges = eg[sample(nrow(eg),58000),]  # Minta az egészségesekből
 
train = rbind(beteg,egeszseges) #tanuló adatbázis újradefiniálása

train[,-c(5,15,16)] = lapply(train[,-c(5,15,16)], as.factor)

# a modell (legjobb arány mellett):

random_vegleges = randomForest(HeartDiseaseorAttack~., heart, mtry = 5 , ntree = 128)

test$predicted = predict(random40, test)  #A klasszifikációs táblázat

table(test$HeartDiseaseorAttack, test$predicted)

# A legjobb klasszifikációt a 58000 egészségest és az összes
# beteget tartalmazó tanuló adatbázison futtatott modell adta.

imp = importance(random_vegleges, type = 2)

# ROC-görbe random forest:

pred_test <- predict(random_vegleges, test, index=2, type= "prob", norm.votes=TRUE, predict.all=FALSE, proximity=FALSE, nodes=FALSE)
pred_test = as.data.frame(pred_test)

pred_test_roc = roc(test$HeartDiseaseorAttack, pred_test$`1`)

plot(pred_test_roc)

auc(pred_test_roc) # FA ROC-görbe alatti területe


roc_fa = ggroc(pred_test_roc, legacy.axes = T, color = "red")

roc_fa + xlab("1-Szenzitivitás")+ylab("Szenzitivitás")+
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color = "green")+
  theme_minimal()+
  ggtitle("A végleges véletlen erdő ROC-görbéje")+    # Random forest ROC-görbéjének ábrázolása
  theme(text = element_text(family = "serif"),
        plot.subtitle = element_text(hjust = 0.5, size = 15),
        plot.title = element_text(hjust = 0.5, size = 19),                     
        axis.title = element_text(size = 15),
        axis.text = element_text(size = 13),
        panel.border = element_rect(fill = NA, colour = "black", size = 1))


#### Neurális Hálózat #### 

library(readxl) #Adatok beolvasásához
library(keras) #Neurális háló építéséhez
library(tensorflow) #Neurális háló építéséhez
library(ROSE) #Adatok arányának megváltoztatásához

heart = read_excel("heartmasolata1.xlsx")

ind = sample(2, nrow(heart), replace = T, prob = c(0.9,0.1)) #Adathalmaz felbontása

trainingtarget = heart[ind == 1, 1]
testtarget = heart[ind == 2, 1]
trainLabels = to_categorical(trainingtarget)  #Eredményváltozók one-hot kódolása
testLabels = to_categorical(testtarget)

#A teszt adatsor átalakítása:

test = heart[ind == 2, 2:21]

test$Income = test$Income - 1
test$Education = test$Education - 1  #Változók első kategóriájának 0-nak kell lennie
test$Age = test$Age - 1
test$GenHlth = test$GenHlth - 1      


test = as.matrix(test)
dimnames(test) = NULL

test[,c(4,14,15)] = normalize(test[,c(4,14,15)]) #Folytonos változók normalizálása
test[,-c(4,14,15)] = lapply(test[,-c(4,14,15)], to_categorical) # kategóriális változók One-hot kódolása

# Arányok megváltoztatása:

heart = read_excel("heartmasolata1.xlsx")
over = over = ovun.sample(HeartDiseaseorAttack~., data = heart, method = "over", N = 459574)

heart$Income = heart$Income - 1
heart$Education = heart$Education - 1  #Változók első kategóriájának 0-nak kell lennie
heart$Age = heart$Age - 1
heart$GenHlth = heart$GenHlth - 1      

heart = as.matrix(heart)
dimnames(heart) = NULL

heart[,c(5,15,16)] = normalize(heart[,c(5,15,16)]) #Folytonos változók normalizálása

ind = sample(2, nrow(heart), replace = T, prob = c(0.9,0.1))

trainingtarget = heart[ind == 1, 1]
testtarget = heart[ind == 2, 1]
trainLabels = to_categorical(trainingtarget) #Eredményváltozó One-hot kódolása
testLabels = to_categorical(testtarget)

training[,-c(4,14,15)] = lapply(training[,-c(4,14,15)], to_categorical) #Magyarázóváltozók one-hot kódolás

#Végső modell:

deep <- keras_model_sequential()
deep %>% 
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 2, activation = "softmax")

#Compile

deep %>% 
  compile(loss = "binary_crossentropy",
    optimizer = optimizer_adam(),   #Költségfüggvény és optimalizáló választása
    metrics = "accuracy")

library(tensorflow)

#Modell training:

fitting <- deep %>% 
  fit(training,
      trainLabels,
      epoch = 40,      #Modell illesztése
      batch_size = 32,
      validation_split =0.20)

summary(deep) #Modell szerkezete

#Konfúziós mátrix:

pred = deep %>%
  predict_classes(test1)
table(Valós = testtarget, Becsült = pred)