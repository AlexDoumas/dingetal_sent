#setwd("~/Desktop")
#getwd()
#jpeg(filename = "rplot.jpg", width = 4, height = 4, units = 'in', res = 300)
###################################
Grammatical = c(0,0,0,0.10)
Jabberwocky = c(0,0,0,0.18)
WordList = c(0,0,0,0.12)
OnlyNPs = c(0,0.0333,0,0.1666)

par(mfrow=c(2,2))

# Panel 1
par(mfg=c(1,1)) # select left-hand figure for drawing
mp1 <- barplot(Grammatical, ylim = c(0, 0.2), names.arg = c("1Hz", "2Hz", "3Hz", "4Hz"),
               col=colors, cex.axis = 1, cex.names = 1.2)            
box()               
mtext("Proportion of units active", side=2, line=2.5, cex=1.2, font=1)
mtext("Grammatical", side=3, line=1.5, cex=1.2, font=1, )

# Panel 2
par(mfg=c(1,2)) # select left-hand figure for drawing
mp2 <- barplot(Jabberwocky, ylim = c(0, 0.2), names.arg = c("1Hz", "2Hz", "3Hz", "4Hz"),
               col=colors, cex.axis = 1, cex.names = 1.2)            
box()               
mtext("Proportion of units active", side=2, line=2.5, cex=1.2, font=1)
mtext("Jabberwocky", side=3, line=1.5, cex=1.2, font=1, )

# Panel 3
par(mfg=c(2,1)) # select left-hand figure for drawing
mp3 <- barplot(WordList, ylim = c(0, 0.2), names.arg = c("1Hz", "2Hz", "3Hz", "4Hz"),
               col=colors, cex.axis = 1, cex.names = 1.2)            
box()               
mtext("Proportion of units active", side=2, line=2.5, cex=1.2, font=1)
mtext("Word List", side=3, line=1.5, cex=1.2, font=1, )

# Panel 4
par(mfg=c(2,2)) # select left-hand figure for drawing
mp3 <- barplot(OnlyNPs, ylim = c(0, 0.2), names.arg = c("1Hz", "2Hz", "3Hz", "4Hz"),
               col=colors, cex.axis = 1, cex.names = 1.2)            
box()               
mtext("Proportion of units active", side=2, line=2.5, cex=1.2, font=1)
mtext("Phrases", side=3, line=1.5, cex=1.2, font=1, )

###############
#dev.off()
