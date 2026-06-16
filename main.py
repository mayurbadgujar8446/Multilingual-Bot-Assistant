from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)

# ── Load model once ───────────────────────────────────────────────────────────
print("Loading model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Model loaded!")

# ── FAQs ─────────────────────────────────────────────────────────────────────
faqs = {
    # --- original (English) + Hinglish + added Hindi + Marathi ---
    # 1
    "what is your name": "I am your BOT ASSISTANT 🤖",
    "tumhara naam kya hai": "Main tumhara BOT ASSISTANT hoon 🤖",
    "आपका नाम क्या है": "मैं आपका BOT ASSISTANT हूँ 🤖",
    "तुमचं नाव काय आहे": "मी तुमचा BOT ASSISTENT आहे 🤖",

    # 2
    "good morning": "Good morning! How can I assist you today?",
    "suprabhat": "Suprabhat! Aaj main aapki kaise madad kar sakta hoon?",
    "सुप्रभात": "सुप्रभात! आज मैं आपकी कैसे मदद कर सकता हूँ?",
    "शुभ प्रभात": "शुभ प्रभात! आज मी तुम्हाला कशी मदत करू शकतो?",

    # 3
    "timetable kab aata hai": "Timetable har semester ke shuru hone se pehle aata hai.",
    "when does the timetable come": "The timetable is usually released before the start of each semester.",
    "समय सारिणी कब आती है": "समय सारिणी आमतौर पर प्रत्येक सेमेस्टर की शुरुआत से पहले जारी की जाती है।",
    "वेळापत्रक कधी येते": "वेळापत्रक सहसा प्रत्येक सेमेस्टरच्या सुरूवातीपूर्वी जाहीर केले जाते.",

    # 4
    "how are you": "I’m doing great! Thanks for asking 😊",
    "tum kaise ho": "Main mast hoon! Tum sunao 😊",
    "आप कैसे हैं": "मैं ठीक हूँ! धन्यवाद पूछने के लिए 😊",
    "तुम्ही कसे आहात": "मी मस्त आहे! तुम्ही सांगा 😊",

    # 5
    "what is ai": "AI stands for Artificial Intelligence. It enables machines to mimic human behavior.",
    "ai kya hai": "AI ka matlab Artificial Intelligence hai. Matlab machine insaan ki tarah soch sakti hai.",
    "AI क्या है": "AI का अर्थ Artificial Intelligence है। यह मशीनों को मानव व्यवहार की नकल करने में सक्षम बनाता है।",
    "AI काय आहे": "AI म्हणजे Artificial Intelligence. हे मशीनला मानवासारखे वर्तन करण्यास सक्षम करते.",

    # 6
    "when are exams conducted": "Exams are usually conducted at the end of each semester 📚",
    "exam kab hote hai": "Pariksha aam taur par semester ke ant me hoti hai 📚",
    "परीक्षाएँ कब होती हैं": "परीक्षाएँ आमतौर पर प्रत्येक सेमेस्टर के अंत में आयोजित की जाती हैं 📚",
    "परीक्षा कधी होतात": "परीक्षा सामान्यतः प्रत्येक सेमेस्टरच्या शेवटी घेतल्या जातात 📚",

    # 7
    "what is placement process": "Placement process includes aptitude tests, group discussions, and interviews.",
    "placement process kya hai": "Placement process me aptitude test, group discussion aur interview hota hai.",
    "प्लेसमेंट प्रक्रिया क्या है": "प्लेसमेंट प्रक्रिया में aptitude test, group discussion और interview होते हैं।",
    "प्लेसमेंट प्रक्रिया काय आहे": "प्लेसमेंट प्रक्रियेत aptitude test, group discussion आणि interview असतात.",

    # ---------------- Additional FAQs (English + Hinglish) with Hindi & Marathi translations ----------------
    # 1 Admissions & Registration
    "how to apply for admission": "Check the college admissions portal for application forms, deadlines, and required documents.",
    "admission ke liye kaise apply karein": "College ke admission portal par jaake form, last date aur zaroori documents check karein.",
    "आवेदन कैसे करें": "कॉलेज के एडमिशन पोर्टल पर जाकर फॉर्म, अंतिम तिथि और आवश्यक दस्तावेज़ देखें।",
    "प्रवेशासाठी अर्ज कसा करावा": "कॉलेजच्या प्रवेश पोर्टलवर जाऊन फॉर्म, शेवटची तारीख आणि आवश्यक कागदपत्रे तपासा.",

    "what documents are required for admission": "Typically: ID proof, mark sheets, passport-size photos, and any entrance scorecard if applicable.",
    "admission ke liye kaunse documents chahiye": "Aam taur par: ID proof, mark sheets, passport-size photos, aur entrance scorecard agar ho to.",
    "प्रवेश के लिए कौन से दस्तावेज़ चाहिए": "आम तौर पर: ID प्रमाण, मार्कशीट, पासपोर्ट साइज फोटो, और यदि लागू हो तो एंट्रेंस स्कोरकार्ड।",
    "प्रवेशासाठी कोणती कागदपत्रे आवश्यक आहेत": "सामान्यतः: ओळख प्रमाणपत्र, मार्कशीट, पासपोर्ट साईझ फोटो आणि एंट्रन्स स्कोरकार्ड (असल्यास).",

    "is entrance exam required": "Some programs require entrance exams; check the specific course admission criteria on the website.",
    "kya entrance exam zaroori hai": "Kuch courses ke liye entrance exam hota hai; course ki admission criteria website par dekhein.",
    "क्या प्रवेश परीक्षा आवश्यक है": "कुछ प्रोग्राम्स के लिए प्रवेश परीक्षा आवश्यक होती है; वेबसाइट पर विशिष्ट कोर्स की शर्तें देखें।",
    "प्रवेश परीक्षा आवश्यक आहे का": "काही कोर्ससाठी प्रवेश परीक्षा आवश्यक असते; संबंधित कोर्सच्या अटी वेबसाइटवर तपासा.",

    "how to pay admission fees": "Use the college online payment portal or pay at the accounts office as instructed by admissions.",
    "admission fees kaise bharein": "College ke online payment portal ka use karein ya accounts office me payment karein, jaise instructions diye gaye hain.",
    "प्रवेश शुल्क कैसे भरें": "कॉलेज के ऑनलाइन पेमेंट पोर्टल का उपयोग करें या एडमिशन के निर्देशानुसार अकाउंट्स ऑफिस में भुगतान करें।",
    "प्रवेश फी कशी भरावी": "कॉलेजच्या ऑनलाइन पेमेंट पोर्टलचा वापर करा किंवा ऍडमिशनच्या सूचनांनुसार अकाउंट्स ऑफिसमध्ये भरा.",

    "last date to submit application": "Deadlines vary by program — always check the official admissions page for exact dates.",
    "application submit karne ki last date kya hai": "Deadline program ke hisab se alag hoti hai — official admissions page par date dekhein.",
    "आवेदन जमा करने की अंतिम तिथि क्या है": "अंतिम तिथियाँ कोर्स अनुसार भिन्न होती हैं — सही तिथि के लिए ऑफिसियल पेज देखें।",
    "अर्ज सबमिट करण्याची शेवटची तारीख कोणती आहे": "तारीख कोर्सनुसार वेगळी असते — नेमकी तारीख अधिकृत पानावर पहा.",

    # 2 Fees & Payments
    "how to pay tuition fees online": "Log in to the student portal → payments → select tuition → pay using card/netbanking/UPI.",
    "tuition fees online kaise bharein": "Student portal me login karein → payments → tuition select karein → card/netbanking/UPI se pay karein.",
    "ट्यूशन फीस ऑनलाइन कैसे भरें": "स्टूडेंट पोर्टल में लॉगिन करें → पेमेंट्स → ट्यूशन चुनें → कार्ड/नेटबैंकिंग/UPI से भुगतान करें।",
    "ट्युशन फी ऑनलाइन कशी भरणी": "स्टुडंट पोर्टल मध्ये लॉगिन करा → पेमेंट्स → ट्युशन निवडा → कार्ड/नेटबँकिंग/UPI वापरून भरा.",

    "what are the fee modes accepted": "Commonly accepted: credit/debit cards, net banking, UPI, and offline challan at bank branches.",
    "kaunse fee payment modes accept hote hain": "Aam taur par: credit/debit card, net banking, UPI, aur offline challan bank me.",
    "कौन-कौन से भुगतानी माध्यम स्वीकार किए जाते हैं": "आम तौर पर: क्रेडिट/डेबिट कार्ड, नेट बैंकिंग, UPI, और ऑफलाइन चालान।",
    "कोणकोणते फी पेमेंट मोड स्वीकारले जातात": "सामान्यतः: क्रेडिट/डेबिट कार्ड, नेटबँकिंग, UPI आणि ऑफलाइन चालान.",

    "is there a fee installment option": "Many colleges offer installment plans — check the accounts office for eligibility and schedule.",
    "kya fee installment option hai": "Kai colleges installment dete hain — accounts office se eligibility aur schedule puchhein.",
    "क्या फीस किस्तों में दे सकते हैं": "कई कॉलेज किस्तों की सुविधा देते हैं — पात्रता व शेड्यूल के लिए अकाउंट्स ऑफिस से पूछें।",
    "फी सुलभ हप्त्यांमध्ये देता येते का": "खूप कॉलेज हप्त्यांची योजना देतात — पात्रता व वेळापत्रकासाठी अकाउंट्स ऑफिसला विचारा.",

    "how to get fee receipt reissued": "Contact the accounts office with payment proof; they can reissue the receipt.",
    "fee receipt phir se kaise milega": "Payment proof ke saath accounts office se sampark karein; vo receipt reissue kar denge.",
    "फीस रसीद फिर से कैसे मिलेगी": "भुगतान का प्रमाण लेकर अकाउंट्स ऑफिस से संपर्क करें; वे रसीद फिर से जारी कर देंगे।",
    "फी रसीद पुन्हा कशी मिळवायची": "पेमेंट पुरावा घेऊन अकाउंट्स ऑफिसला संपर्क करा; ते रसीद पुनरुत्पादित करतील.",

    "what is the late fee policy": "Late fees are charged after the due date — refer to the fee policy on the college website for rates.",
    "late fee policy kya hai": "Due date ke baad late fees lagti hai — college website par fee policy dekhein.",
    "लेट फीस नीति क्या है": "नियत तिथि के बाद लेट फीस लागू होती है — दरों के लिए कॉलेज वेबसाइट पर देखें।",
    "लेट फी धोरण काय आहे": "नियत तारीखेनंतर लेट फी लागू होते — दरांसाठी कॉलेज वेबसाइट पहा.",

    # 3 Scholarships & Financial Aid
    "how to apply for scholarships": "Check the scholarships page, fill the online application, and submit supporting documents before the deadline.",
    "scholarship ke liye kaise apply karein": "Scholarships page dekhein, online form bharein aur documents deadline se pehle submit karein.",
    "छात्रवृत्ति के लिए कैसे आवेदन करें": "स्कॉलरशिप पेज देखें, ऑनलाइन फॉर्म भरें और आवश्यक दस्तावेज़ समय से पहले जमा करें।",
    "शिष्यवृत्ती साठी कशी अर्ज करावी": "शिष्यवृत्ती पृष्ठ पहा, ऑनलाइन फॉर्म भरा आणि दस्तऐवज वेळेपूर्वी सबमिट करा.",

    "what scholarships are available": "Merit-based, need-based, sports, and category-specific scholarships are commonly offered.",
    "kaunse scholarships available hain": "Merit-based, need-based, sports aur category-specific scholarships aam taur par milti hain.",
    "कौन-कौन सी स्कॉलरशिप उपलब्ध हैं": "मेरिट, आवश्यकता, खेल और श्रेणी-विशिष्ट स्कॉलरशिप आमतौर पर उपलब्ध होती हैं।",
    "कोणकोणत्या शिष्यवृत्ती उपलब्ध आहेत": "मेरिट-आधारित, गरज-आधारित, क्रीडा आणि वर्ग-विशिष्ट शिष्यवृत्ती साधारणपणे असतात.",

    "scholarship renewal criteria": "Most require minimum GPA/attendance and submission of renewal form with documents.",
    "scholarship renew kaise hoti hai": "Aksar minimum GPA/attendance aur renewal form + documents chahiye hote hain.",
    "छात्रवृत्ति नवीनीकरण के मानदंड क्या हैं": "अधिकांश में न्यूनतम GPA/उपस्थिति और नवीनीकरण फॉर्म के साथ दस्तावेज़ आवश्यक होते हैं।",
    "शिष्यवृत्ती नवीनीकरण निकष काय आहेत": "अधिकांशाला किमान GPA/हजेरी आणि नवीनीकरण फॉर्म + कागदपत्रे लागतात.",

    "is financial aid available for international students": "Some colleges offer aid for international students — check international admissions or financial aid office.",
    "kya international students ke liye financial aid hai": "Kuch colleges international students ke liye aid dete hain — international admissions/financial aid office se puchhein.",
    "क्या अंतरराष्ट्रीय छात्रों के लिए वित्तीय सहायता है": "कुछ कॉलेज अंतरराष्ट्रीय छात्रों को सहायता प्रदान करते हैं — अंतरराष्ट्रीय प्रवेश या वित्तीय सहायता कार्यालय से पूछें।",
    "जागतिक विद्यार्थ्यांसाठी आर्थिक मदत उपलब्ध आहे का": "काही कॉलेज जागतिक विद्यार्थ्यांसाठी मदत देतात — आंतरराष्ट्रीय प्रवेश/आर्थिक मदत कार्यालयात विचार करा.",

    # 4 Academic Calendar & Timetable
    "where to find the academic calendar": "Academic calendar is usually on the college website under Academics/Calendar.",
    "academic calendar kahaan milega": "College website ke Academics/Calendar section me academic calendar hota hai.",
    "अकादमिक कैलेंडर कहाँ मिलेगा": "अकादमिक कैलेंडर सामान्यतः कॉलेज की वेबसाइट के Academics/Calendar सेक्शन में होता है।",
    "अकॅडमिक कॅलेंडर कुठे मिळेल": "अकॅडमिक कॅलेंडर सामान्यतः कॉलेजच्या वेबसाइटवर Academics/Calendar मध्ये असतो.",

    "how often is timetable updated": "Timetables update before each semester or if administrative changes occur.",
    "timetable kitni baar update hota hai": "Har semester ke pehle ya administrative changes par timetable update hota hai.",
    "समय सारिणी कितनी बार अपडेट होती है": "समय सारिणी प्रत्येक सेमेस्टर से पहले या प्रशासनिक बदलाव होने पर अपडेट की जाती है।",
    "वेळापत्रक किती वेळा अपडेट होते": "वेळापत्रक सहसा प्रत्येक सेमेस्टरपूर्वी किंवा प्रशासनिक बदलांनुसार अपडेट केले जाते.",

    "how to download timetable": "Login to student portal → timetable/download or check notices for PDF links.",
    "timetable kaise download karein": "Student portal me login karein → timetable/download ya notices me PDF link dekhein.",
    "समय सारिणी कैसे डाउनलोड करें": "स्टूडेंट पोर्टल में लॉगिन करें → timetable/download या नोटिस में PDF लिंक देखें।",
    "वेळापत्रक कसे डाउनलोड करावे": "स्टुडंट पोर्टलमध्ये लॉगिन करा → timetable/download किंवा नोटिसमधील PDF लिंक तपासा.",

    # 5 Attendance & Leave
    "what is the minimum attendance required": "Usually 75% attendance is required; check your college’s specific attendance rules.",
    "minimum attendance kitni chahiye": "Aksar 75% attendance required hoti hai; college ki specific policy dekhein.",
    "न्यूनतम उपस्थिति कितनी चाहिए": "आम तौर पर 75% उपस्थिति आवश्यक होती है; अपने कॉलेज की नीति देखें।",
    "किमान हजेरी किती लागते": "सामान्यतः 75% हजेरी आवश्यक असते; कॉलेजची धोरण पाहा.",

    "how to apply for leave": "Submit leave application via student portal or to the department office with supporting documents.",
    "leave kaise apply karein": "Student portal se ya department office me supporting documents ke saath leave application dein.",
    "छुट्टी के लिए कैसे आवेदन करें": "छुट्टी एप्लिकेशन स्टूडेंट पोर्टल के माध्यम से या विभाग कार्यालय में दस्तावेज़ों के साथ जमा करें।",
    "सुट्टीसाठी कशी अर्ज करावी": "स्टुडंट पोर्टलद्वारे किंवा विभाग कार्यालयात पुराव्यांसहित सुट्टी अर्ज सबमिट करा.",

    "what counts as medical leave": "Medical leave requires a doctor's certificate and prior/intimation to class-in-charge where possible.",
    "medical leave kya maana jata hai": "Medical leave ke liye doctor ka certificate aur class-in-charge ko pehle batana zaroori hota hai.",
    "मेडिकल लीव किसे माना जाता है": "मेडिकल लीव के लिए डॉक्टर का प्रमाण-पत्र और संभव हो तो क्लास-इन-चार्ज को सूचित करना आवश्यक है।",
    "वैद्यकीय सुट्टीला काय मानले जाते": "वैद्यकीय सुट्टीसाठी डॉक्टरचे प्रमाणपत्र आणि शक्य असल्यास क्लास-इन-चार्जला पूर्वसूचना आवश्यक आहे.",

    # 6 Exams & Results
    "how to register for exams": "Exam registration is via the student portal during registration windows; check notifications for dates.",
    "exam ke liye kaise register karein": "Exam registration student portal se hoti hai; notifications me dates dekhein.",
    "परीक्षा के लिए कैसे रजिस्टर करें": "परीक्षा पंजीकरण स्टूडेंट पोर्टल के माध्यम से होता है; तारीखों के लिए नोटिफिकेशन्स देखें।",
    "परीक्षेसाठी नोंदणी कशी करावी": "परीक्षेची नोंदणी स्टुडंट पोर्टलवर होते; तारीखांसाठी सूचनांचा अवलंब करा.",

    "when will results be declared": "Result dates vary; watch the exam cell notices or student portal for announcements.",
    "results kab declare hote hain": "Result ki dates alag hoti hain; exam cell notices aur student portal dekhte rahen.",
    "परिणाम कब घोषित होंगे": "परिणाम की तारीखें अलग-अलग होती हैं; परीक्षा सेल नोटिस और स्टूडेंट पोर्टल देखें।",
    "उपसंहार कधी जाहीर होतात": "परिणामाच्या तारखा वेगवेगळ्या असतात; परीक्षा सेल नोटिस व स्टुडंट पोर्टल पहा.",

    "how to apply for revaluation": "Fill the revaluation form within the specified window and pay the revaluation fee as instructed.",
    "revaluation ke liye kaise apply karein": "Nirdharit time window me revaluation form bharein aur fee jama karein.",
    "पुनर्मूल्यांकन के लिए कैसे आवेदन करें": "नियत समय सीमा में पुनर्मूल्यांकन फॉर्म भरें और शुल्क जमा करें।",
    "पुनर्मुल्यांकनासाठी कसे अर्ज करावे": "निर्दिष्ट वेळेमध्ये पुनर्मुल्यांकन फॉर्म भरा आणि फी भरा.",

    "what is supplementary exam": "Supplementary (or backlog) exam lets students reattempt failed papers — check exam rules and schedule.",
    "supplementary exam kya hota hai": "Supplementary exam se failed papers dobara attempt kiye ja sakte hain — exam rules dekhein.",
    "पूरक परीक्षा क्या होती है": "पूरक परीक्षा से असफल पेपर्स को पुनः प्रयत्न करने का मौका मिलता है — नियम देखें।",
    "पूरक परीक्षा म्हणजे काय": "पूरक परीक्षेमुळे अपयशी पेपर्स परत देण्याची संधी मिळते — परीक्षा नियम पहा.",

    # 7 Grading, Transcripts & Certificates
    "how is cgpa calculated": "CGPA is calculated based on weighted grade points across courses as per university rules.",
    "cgpa kaise calculate hota hai": "CGPA courses ke weighted grade points ke hisab se university rules ke mutabik calculate hota hai.",
    "CGPA कैसे हिसाब किया जाता है": "CGPA विश्वविद्यालय के नियमों के अनुसार पाठ्यक्रमों के weighted grade points पर आधारित होता है।",
    "CGPA कसा मोजला जातो": "CGPA विद्यापीठाच्या नियमांनुसार कोर्सचे weighted grade points वरून मोजले जाते.",

    "how to get a transcript": "Apply to the registrar/records office or use the student portal; there may be a processing fee.",
    "transcript kaise milega": "Registrar/records office me apply karein ya student portal se; processing fee ho sakti hai.",
    "ट्रांसक्रिप्ट कैसे प्राप्त करें": "रजिस्ट्रार/रेकॉर्ड्स ऑफिस में आवेदन करें या स्टूडेंट पोर्टल का उपयोग करें; प्रक्रिया शुल्क हो सकता है।",
    "ट्रान्सक्रिप्ट कशी मिळवायची": "रजिस्ट्रार/रेकॉर्ड्स ऑफिसमध्ये अर्ज करा किंवा स्टुडंट पोर्टल वापरा; प्रक्रियेनुसार फी लागू असू शकते.",

    "how long to process certificates": "Processing time varies (usually 1–4 weeks); check with the records office for exact timelines.",
    "certificates process hone me kitna time lagta hai": "Processing time alag hota hai (aam taur par 1–4 weeks); records office se puchhein.",
    "प्रमाण-पत्रों को प्रक्रिया करने में कितना समय लगता है": "प्रक्रिया समय अलग होता है (आम तौर पर 1–4 सप्ताह); सटीक समय के लिए रिकॉर्ड्स ऑफिस से पूछें।",
    "प्रमाणपत्र प्रक्रिया होण्यासाठी किती वेळ लागतो": "प्रक्रिया वेळ वेगवेगळी असते (साधारणतः 1–4 आठवडे); नेमका कालावधी रेकॉर्ड्स ऑफिसला विचारा.",

    "how to get migration certificate": "Apply to the academic/records office with necessary IDs and fees; follow the college process.",
    "migration certificate kaise milega": "Academic/records office me apply karein, ID aur fees lekar; college ki process follow karein.",
    "माइग्रेशन सर्टिफिकेट कैसे लें": "आवश्यक ID और फीस के साथ अकादमिक/रेकॉर्ड्स ऑफिस में आवेदन करें; कॉलेज प्रक्रिया का पालन करें।",
    "मायग्रेशन सर्टिफिकेट कसे मिळवायचे": "आवश्यक ओळखपत्र व फी घेऊन अकॅडमिक/रेकॉर्ड्स ऑफिसमध्ये अर्ज करा; कॉलेजची प्रक्रिया पाळा.",

    # 8 Library & Learning Resources
    "how to join the library": "Library membership is automatic for enrolled students; visit the library with your student ID to activate it.",
    "library me kaise join karein": "Enrollment par library membership aam taur par automatic hoti hai; student ID lekar library jayen activation ke liye.",
    "लाइब्रेरी में कैसे शामिल हों": "नामांकित छात्रों के लिए लाइब्रेरी सदस्यता सामान्यतः स्वतः होती है; सक्रिय करने के लिए अपनी छात्र ID लेकर लाइब्रेरी जाएं।",
    "लायब्ररीमध्ये कसे सामील व्हावे": "नोंदणी केलेल्या विद्यार्थ्यांसाठी लायब्ररी सदस्यत्व सामान्यतः आपोआप असते; सक्रिय करण्यासाठी स्टुडंट ID घेऊन लायब्ररीमध्ये जा.",

    "library opening hours": "Library hours vary; check the library webpage or noticeboard for current timings.",
    "library ki timing kya hai": "Library timings alag hoti hain; library webpage ya noticeboard par dekhein.",
    "लाइब्रेरी का खुलने का समय क्या है": "लाइब्रेरी का समय बदलता है; वर्तमान समय के लिए लाइब्रेरी वेबपेज या नोटिसबोर्ड देखें।",
    "लायब्ररीच्या उघडण्याची वेळ काय आहे": "लायब्ररीच्या वेळा बदलतात; चालू वेळेसाठी लायब्ररी वेबसाइट किंवा नोटिसबोर्ड पहा.",

    "how to renew books": "Use the library portal or visit the counter with your book to renew if no hold exists.",
    "books kaise renew karein": "Library portal se ya counter par jaake renew karein agar book par hold na ho.",
    "पुस्तकें कैसे नवीनीकरण करें": "यदि कोई होल्ड नहीं है तो लाइब्रेरी पोर्टल का उपयोग करें या काउंटर पर जाकर पुस्तक नवीनीकरण करवाएं।",
    "पुस्तके कसे नवीनीकरण करावीत": "जर होल्ड नसेल तर लायब्ररी पोर्टल वापरा किंवा काउंटरवर जाऊन पुस्तके नवीनीकरण करा.",

    "what is the fine for late return": "Late fines are charged per day per book; see library fine policy for specific rates.",
    "late return ka fine kitna hai": "Per day per book fine lagta hai; specific rates ke liye library fine policy dekhein.",
    "देर से लौटाने पर जुर्माना कितना है": "प्रति पुस्तक प्रति दिन लेट फाइन लगता है; दरों के लिए लाइब्रेरी फाइन नीति देखें।",
    "उशीराने परत केल्यास दंड किती आहे": "प्रति पुस्तक प्रतिदिन लेट फाइन लागतो; विशिष्ट दरांसाठी लायब्ररीची फाइन धोरण पहा.",

    "how to access e-resources": "Login to the library e-resources portal using student credentials for journals, ebooks, and databases.",
    "e-resources kaise access karein": "Student credentials se library e-resources portal me login karke journals, ebooks aur databases access karein.",
    "ई-रिसोर्सेज़ कैसे एक्सेस करें": "जर्नल, ईबुक और डेटाबेस के लिए स्टूडेंट क्रेडेंशियल से लाइब्रेरी e-resources पोर्टल में लॉगिन करें।",
    "ई-संसाधन कसे प्रवेश करावे": "जर्नल्स, ई-बुक्स आणि डेटाबेससाठी स्टुडंट क्रेडेन्शियलने लायब्ररी ई-रिसोर्स पोर्टलमध्ये लॉगिन करा.",

    # 9 Hostels & Accommodation
    "how to apply for hostel": "Apply via hostel portal or admission office; application windows are announced before semester start.",
    "hostel ke liye kaise apply karein": "Hostel portal ya admission office se apply karein; application window semester se pehle announce hoti hai.",
    "हॉस्टल के लिए कैसे आवेदन करें": "हॉस्टल पोर्टल या एडमिशन ऑफिस के माध्यम से आवेदन करें; आवेदन विंडो सेमेस्टर शुरू होने से पहले घोषित की जाती है।",
    "हॉस्टेलसाठी कसा अर्ज करावा": "हॉस्टेल पोर्टल किंवा प्रवेश कार्यालयाद्वारे अर्ज करा; अर्ज विंडो सेमेस्टरपूर्वी जाहीर केली जाते.",

    "what is hostel allotment criteria": "Allotment may consider merit, distance, special quotas, and availability.",
    "hostel allotment criteria kya hai": "Allotment merit, distance, special quotas aur availability par depend kar sakti hai.",
    "हॉस्टल आवंटन के मानदंड क्या हैं": "आवंटन मेऱिट, दूरी, विशेष कोटेदारियां, और उपलब्धता के आधार पर हो सकता है।",
    "हॉस्टेल वाटप निकष काय आहेत": "वाटप मेरिट, अंतर, विशेष कोटा आणि उपलब्धतेवर अवलंबून असते.",

    "can I change hostel room": "Room changes allowed subject to availability and approval from the hostel office.",
    "kya room change ho sakta hai": "Room change availability aur hostel office ki permission par hota hai.",
    "क्या मैं हॉस्टल का कमरा बदल सकता हूँ": "कक्ष परिवर्तन उपलब्धता और हॉस्टल ऑफिस की अनुमति पर निर्भर करता है।",
    "मी हॉस्टेल रूम बदलू शकतो का": "रूम बदल उपलब्धता आणि हॉस्टेल कार्यालयाच्या परवानगीवर अवलंबून आहे.",

    "what are hostel rules": "Hostel rules cover curfew, visitors, cleanliness, noise, and safety — read the hostel handbook.",
    "hostel rules kya hain": "Hostel rules me curfew, visitors, safai, noise aur safety jaise points hote hain — hostel handbook padhein.",
    "हॉस्टल नियम क्या हैं": "हॉस्टल नियमों में कर्फ्यू, आगंतुक, स्वच्छता, शोर और सुरक्षा शामिल हैं — हॉस्टल हैंडबुक पढ़ें।",
    "हॉस्टेल नियम काय आहेत": "हॉस्टेलचे नियम कर्फ्यू, पाहुणे, स्वच्छता, आवाज आणि सुरक्षा यांचा समावेश करतात — हॉस्टेल हँडबुक वाचा.",

    "how to pay mess charges": "Mess charges are billed monthly; pay via hostel portal or accounts office as instructed.",
    "mess charges kaise bharein": "Mess charges mahine bhar ke aate hain; hostel portal ya accounts office se pay karein.",
    "मेस शुल्क कैसे भरें": "मेस शुल्क मासिक बिल के रूप में आता है; निर्देशानुसार हॉस्टेल पोर्टल या अकाउंट्स ऑफिस में भुगतान करें।",
    "मेसे शुल्क कशी भरणी": "मेसे शुल्क मासिक आले जाते; सूचनांनुसार हॉस्टेल पोर्टल किंवा अकाउंट्स ऑफिसमध्ये भरा.",

    # 10 Canteen, Facilities & Campus Life
    "canteen timings and menu": "Canteen timings and menu are posted on the notice board and often on the college app.",
    "canteen ki timing aur menu": "Canteen timings aur menu notice board ya college app par available hote hain.",
    "कैंटीन का समय और मेनू क्या है": "कैंटीन का समय और मेनू नोटिस बोर्ड और कॉलेज ऐप पर पोस्ट किया जाता है।",
    "कॅन्टीनची वेळ व मेनू काय आहे": "कॅन्टीनची वेळ व मेनू नोटिस बोर्डवर आणि बहुतेकदा कॉलेज अॅपवर पोस्ट केलेले असतात.",

    "are vegetarian options available": "Yes, most canteens offer vegetarian options; check menu or ask canteen staff.",
    "kya vegetarian options milte hain": "Haan, adhiktar canteens me vegetarian options milte hain; menu dekhein ya staff se puchhein.",
    "क्या शाकाहारी विकल्प उपलब्ध हैं": "हां, अधिकांश कैंटीन में शाकाहारी विकल्प होते हैं; मेनू देखें या स्टाफ से पूछें।",
    "शाकाहारी पर्याय उपलब्ध आहेत का": "हो, बहुतेक कॅन्टीनमध्ये शाकाहारी पर्याय असतात; मेनू पहा किंवा स्टाफला विचारा.",

    "how to book sports facilities": "Contact the sports office or use the sports booking portal to reserve courts/fields.",
    "sports facilities kaise book karein": "Sports office se sampark karein ya sports booking portal se courts/fields reserve karein.",
    "खेल सुविधाएँ कैसे बुक करें": "स्पोर्ट्स ऑफिस से संपर्क करें या कोर्ट/फील्ड आरक्षित करने के लिए स्पोर्ट्स बुकिंग पोर्टल का उपयोग करें।",
    "क्रीडा सुविधा कशी बुक करावी": "क्रीडा कार्यालयाशी संपर्क करा किंवा कोर्ट/फिल्ड राखीव करण्यासाठी बुकिंग पोर्टल वापरा.",

    "is there a gym on campus": "Many colleges have a campus gym; check gym registration requirements and timings.",
    "campus par gym hai kya": "Kai colleges me gym hota hai; gym registration requirements aur timings dekhein.",
    "क्या कैंपस पर जिम है": "कई कॉलेजों में कैंपस जिम होता है; पंजीकरण आवश्यकताएं और समय देखें।",
    "कॅम्पसवर जिम आहे का": "खूप पुढील कॉलेजमध्ये जिम असते; नोंदणी अटी व वेळा तपासा.",

    # 11 IT & Connectivity
    "how to get college wifi access": "Register your device on the IT portal with student credentials to get wifi access.",
    "college wifi access kaise milega": "IT portal par student credentials se device register karke wifi access paayen.",
    "कॉलेज वाईफाई कैसे प्राप्त करें": "आईटी पोर्टल पर स्टूडेंट क्रेडेंशियल से अपना डिवाइस रजिस्टर करें ताकि वाईफाई एक्सेस मिले।",
    "कॉलेज वायफाय ऍक्सेस कसा मिळवायचा": "आयटी पोर्टलवर स्टुडंट क्रेडेन्शियलने आपले डिवाइस नोंदवा अधिक वायफाय मिळेल.",

    "how to reset student email password": "Use the IT self-service password reset or contact the IT helpdesk with ID proof.",
    "email password kaise reset karein": "IT self-service password reset ka use karein ya IT helpdesk ko ID proof ke saath contact karein.",
    "स्टूडेंट ईमेल पासवर्ड कैसे रीसेट करें": "आईटी सेल्फ-सर्विस पासवर्ड रीसेट का उपयोग करें या आईटी हेल्पडेस्क को ID प्रमाण के साथ संपर्क करें।",
    "ईमेल पासवर्ड रीसेट कसा करायचा": "आयटी सेल्फ-सर्विस पासवर्ड रीसेट वापरा किंवा आयडी पुराव्यासह आयटी हेल्पडेस्कला संपर्क करा.",

    "where to report it issues": "Submit a ticket to the IT helpdesk or contact them by phone/email as listed on the website.",
    "it problems kahan report karein": "IT helpdesk me ticket submit karein ya website par listed phone/email se contact karein.",
    "आईटी समस्याएँ कहाँ रिपोर्ट करें": "आईटी हेल्पडेस्क को टिकट सबमिट करें या वेबसाइट पर सूचीबद्ध फोन/ईमेल से संपर्क करें।",
    "आयटी समस्या कुठे नोंदवायच्या": "आयटी हेल्पडेस्कला तिकीट सबमिट करा किंवा वेबसाइटवरील फोन/ईमेलवर संपर्क करा.",

    "how to access the LMS": "Login to the LMS with your college credentials; course pages, assignments, and materials are posted there.",
    "LMS kaise access karein": "College credentials se LMS me login karein; course pages aur assignments waha milenge.",
    "LMS कैसे एक्सेस करें": "कॉलेज क्रेडेंशियल से LMS में लॉगिन करें; कोर्स पेज, असाइनमेंट और सामग्रियाँ वहां मिलेंगी।",
    "LMS कसे प्रवेश करावे": "कॉलेज क्रेडेन्शियलने LMS मध्ये लॉगिन करा; कोर्स पेजेस, असाइनमेंट व सामग्री तिथे मिळतील.",

    # 12 Labs, Equipment & Safety
    "how to book lab time": "Lab bookings are managed by departments — check lab schedule or book via department portal.",
    "lab ka time kaise book karein": "Departments lab schedule manage karte hain — department portal par check karein.",
    "लैब समय कैसे बुक करें": "लैब बुकिंग विभाग द्वारा प्रबंधित की जाती है — लैब शेड्यूल देखें या विभाग पोर्टल से बुक करें।",
    "लॅब वेळ कशी बुक करावी": "विभागांकडून लॅब वेळ व्यवस्थापित केला जातो — शेड्यूल तपासा किंवा विभाग पोर्टलवर बुक करा.",

    "what safety gear is required in labs": "Wear lab coats, goggles, and follow lab safety protocols; use equipment only after training.",
    "lab me safety gear kya chahiye": "Lab coat, goggles pehnen aur safety protocols follow karein; training ke baad hi equipment use karein.",
    "लैब में किस सुरक्षा उपकरण की आवश्यकता है": "लैब कोट, गॉगल्स पहनें और लैब सुरक्षा प्रोटोकॉल का पालन करें; प्रशिक्षण के बाद ही उपकरणों का उपयोग करें।",
    "लॅबमध्ये कोणते सुरक्षा उपकरण आवश्यक आहेत": "लॅब कोट, गॉगल्स घाला आणि सुरक्षा नियम पाळा; प्रशिक्षणानंतरच उपकरणे वापरा.",

    "how to report lab breakage": "Inform the lab in-charge and file a report; follow department policy for repairs/charges.",
    "lab breakage kaise report karein": "Lab in-charge ko batayein aur report file karein; department policy follow karein repairs/charges ke liye.",
    "लैब टूट-फूट की रिपोर्ट कैसे करें": "लैब इन्चार्ज को सूचित करें और रिपोर्ट दर्ज करें; मरम्मत/शुल्क के लिए विभाग की नीति का पालन करें।",
    "लॅब बिघाड कसा नोंदवायचा": "लॅब इंचार्जला सांगा आणि अहवाल दाखल करा; दुरुस्ती/शुल्कांसाठी विभागाच्या धोरणाचे पालन करा.",

    # 13 Projects, Internships & Research
    "how to choose a project guide": "Discuss interests with faculty and check availability; your department will provide allocation details.",
    "project guide kaise choose karein": "Faculty se apni interests discuss karein aur availability check karein; department allocation details denge.",
    "प्रोजेक्ट गाइड कैसे चुनें": "फैकल्टी से अपनी रुचि पर चर्चा करें और उपलब्धता जांचें; विभाग आवंटन विवरण प्रदान करेगा।",
    "प्रोजेक्ट गाईड कसा निवडावा": "प्राध्यापकांशी आपली आवड सांगा व उपलब्धता तपासा; विभाग वाटप तपशील देईल.",

    "are internships mandatory": "Some programs require internships for credits; check your course curriculum.",
    "kya internships mandatory hain": "Kuch programs me internships credits ke liye zaroori hoti hain; curriculum check karein.",
    "क्या इंटर्नशिप अनिवार्य हैं": "किसी-किसी प्रोग्राम में क्रेडिट के लिए इंटर्नशिप अनिवार्य होती हैं; अपने पाठ्यक्रम की जांच करें।",
    "इंटर्नशिप आवश्यक आहेत का": "काही कार्यक्रमांमध्ये क्रेडिटसाठी इंटर्नशिप आवश्यक असू शकते; अभ्यासक्रम तपासा.",

    "how to apply for internships": "Search portals, placement cell announcements, or approach companies directly with your resume.",
    "internship ke liye kaise apply karein": "Placement cell announcements, job portals ya directly companies ko resume bhejein.",
    "इंटर्नशिप के लिए कैसे आवेदन करें": "प्लेसमेंट सेल घोषणाएं, नौकरी पोर्टल या सीधे कंपनियों को रिज्यूमे भेजकर आवेदन करें।",
    "इंटर्नशिपसाठी कशी अर्ज करावी": "प्लेसमेंट सेलच्या जाहिराती, नोकरी पोर्टल किंवा थेट कंपन्यांना तुमचे रेज्युमे पाठवा.",

    "how are internship evaluations done": "Typically via company feedback, mentor report, and a final submission/presentation to the college.",
    "internship ka evaluation kaise hota hai": "Company feedback, mentor report aur college me final submission/presentation se hota hai.",
    "इंटर्नशिप का मूल्यांकन कैसे होता है": "आमतौर पर कंपनी प्रतिक्रिया, मेंटर रिपोर्ट और कॉलेज में अंतिम सादरीकरण/रिपोर्ट के माध्यम से।",
    "इंटर्नशिपचे मूल्यमापन कसे केले जाते": "सामान्यतः कंपनी फीडबॅक, मेंटर अहवाल आणि कॉलेजमधील अंतिम सादरीकरण/सबमिशनद्वारे.",

    # 14 Placements & Careers
    "how to register for campus placements": "Register on the placement cell portal and attend orientation sessions that prepare students.",
    "campus placements ke liye kaise register karein": "Placement cell portal me register karein aur orientation sessions attend karein.",
    "कैंपस प्लेसमेंट के लिए कैसे रजिस्टर करें": "प्लेसमेंट सेल पोर्टल पर रजिस्टर करें और तैयारी हेतु ओरिएंटेशन सत्रों में भाग लें।",
    "कॅम्पस प्लेसमेंट साठी कसा नोंदणी करावे": "प्लेसमेंट सेल पोर्टलवर नोंदणी करा आणि तयारी सत्र उपस्थित रहा.",

    "what is placement eligibility": "Eligibility may include minimum CGPA, attendance, and no active disciplinary actions.",
    "placement eligibility kya hoti hai": "Eligibility minimum CGPA, attendance aur disciplinary status par depend karti hai.",
    "प्लेसमेंट के लिए पात्रता क्या है": "पात्रता में न्यूनतम CGPA, उपस्थिति और कोई सक्रिय अनुशासनात्मक कार्रवाई न होना शामिल हो सकता है।",
    "प्लेसमेंट पात्रता काय आहे": "पात्रतेमध्ये किमान CGPA, हजेरी आणि कोणतीही शिस्तभंग कारवाई नसणे अपेक्षित असू शकते.",


    # ------------------ NEW ADDITIONS START HERE -------------------

    # 15 Student Life & Activities
    "how to join a student club": "Visit the student affairs office or attend the club fair at the start of the semester to sign up.",
    "student club kaise join karein": "Student affairs office jayein ya semester ke shuru me hone wale club fair me sign up karein.",
    "स्टूडेंट क्लब कैसे ज्वाइन करें": "स्टूडेंट अफेयर्स ऑफिस जाएँ या सेमेस्टर की शुरुआत में होने वाले क्लब फेयर में साइन अप करें।",
    "विद्यार्थी क्लबमध्ये कसे सामील व्हावे": "विद्यार्थी व्यवहार कार्यालयाला भेट द्या किंवा सेमेस्टरच्या सुरुवातीला होणाऱ्या क्लब फेअरमध्ये नोंदणी करा.",

    "where can I find information about events": "Check the college's official social media, website, and notice boards for event announcements.",
    "events ki information kahan milegi": "College ke official social media, website, aur notice board par events ki announcements dekhein.",
    "इवेंट्स की जानकारी कहाँ मिलेगी": "कॉलेज के ऑफिसियल सोशल मीडिया, वेबसाइट और नोटिस बोर्ड पर इवेंट्स की घोषणाएं देखें।",
    "कार्यक्रमांची माहिती कुठे मिळेल": "कॉलेजच्या अधिकृत सोशल मीडिया, वेबसाइट आणि नोटीस बोर्डवर कार्यक्रमांच्या घोषणा पहा.",

    "is there a student council": "Yes, the student council represents the student body; check the website for their contact details.",
    "kya student council hai": "Haan, student council students ko represent karti hai; unke contact details website par dekhein.",
    "क्या स्टूडेंट काउंसिल है": "हाँ, स्टूडेंट काउंसिल छात्रों का प्रतिनिधित्व करती है; उनके संपर्क विवरण वेबसाइट पर देखें।",
    "विद्यार्थी परिषद आहे का": "हो, विद्यार्थी परिषद विद्यार्थ्यांचे प्रतिनिधित्व करते; त्यांच्या संपर्काची माहिती वेबसाइटवर पहा.",

    "how to get a student ID card": "A student ID card is issued at the time of admission or during registration; contact the administration office for a new one.",
    "student ID card kaise milega": "Student ID card admission ke time ya registration ke dauran milta hai; naya banwane ke liye administration office se contact karein.",
    "स्टूडेंट आईडी कार्ड कैसे मिलेगा": "स्टूडेंट आईडी कार्ड एडमिशन के समय या रजिस्ट्रेशन के दौरान दिया जाता है; नया बनवाने के लिए प्रशासन कार्यालय से संपर्क करें।",
    "विद्यार्थी ओळखपत्र कसे मिळवायचे": "विद्यार्थी ओळखपत्र प्रवेशाच्या वेळी किंवा नोंदणीच्या दरम्यान दिले जाते; नवीन बनवण्यासाठी प्रशासन कार्यालयाशी संपर्क साधा.",

    # 16 Health & Safety
    "where is the medical center": "The campus medical center is located [location]. It provides first aid and basic medical consultation.",
    "medical center kahan hai": "Campus medical center [location] par hai. Yahan first aid aur basic medical consultation milta hai.",
    "मेडिकल सेंटर कहाँ है": "कैंपस मेडिकल सेंटर [स्थान] पर स्थित है। यह प्राथमिक उपचार और सामान्य चिकित्सा परामर्श प्रदान करता है।",
    "वैद्यकीय केंद्र कुठे आहे": "कॅम्पस वैद्यकीय केंद्र [स्थान] येथे आहे. येथे प्रथमोपचार आणि सामान्य वैद्यकीय सल्ला उपलब्ध आहे.",

    "what to do in an emergency": "Contact the campus security office immediately at [phone number] or dial the emergency hotline.",
    "emergency me kya karein": "Emergency me campus security office ko turant [phone number] par contact karein ya emergency hotline par dial karein.",
    "आपातकाल में क्या करें": "आपातकाल में तुरंत कैंपस सुरक्षा कार्यालय को [फोन नंबर] पर संपर्क करें या आपातकालीन हॉटलाइन पर डायल करें।",
    "आणीबाणीत काय करावे": "आणीबाणीत कॅम्पस सुरक्षा कार्यालयाशी [फोन नंबर] वर त्वरित संपर्क साधा किंवा आपत्कालीन हॉटलाइनवर फोन करा.",

    "is counseling services available": "Yes, confidential counseling services are available for students. Book an appointment through the student wellness portal.",
    "kya counseling services hain": "Haan, students ke liye confidential counseling services available hain. Student wellness portal se appointment book karein.",
    "क्या काउंसलिंग सेवाएं उपलब्ध हैं": "हाँ, छात्रों के लिए गोपनीय काउंसलिंग सेवाएं उपलब्ध हैं। स्टूडेंट वेलनेस पोर्टल के माध्यम से अपॉइंटमेंट बुक करें।",
    "समुपदेशन सेवा उपलब्ध आहे का": "हो, विद्यार्थ्यांसाठी गोपनीय समुपदेशन सेवा उपलब्ध आहेत. विद्यार्थी कल्याण पोर्टलद्वारे अपॉइंटमेंट बुक करा.",

    # 17 Administrative Services
    "how to update my address": "Fill out the address change form on the student portal and submit it to the registrar's office with ID proof.",
    "mera address kaise update karein": "Student portal par address change form bharein aur ID proof ke saath registrar ke office me submit karein.",
    "मेरा पता कैसे अपडेट करें": "स्टूडेंट पोर्टल पर पता बदलने का फॉर्म भरें और आईडी प्रमाण के साथ रजिस्ट्रार कार्यालय में जमा करें।",
    "माझा पत्ता कसा अपडेट करायचा": "विद्यार्थी पोर्टलवर पत्ता बदलायचा फॉर्म भरा आणि ओळखपत्रासह रजिस्ट्रारच्या कार्यालयात जमा करा.",

    "how to get an official letter": "Submit a request for an official letter, such as a bonafide certificate, through the student portal or to the registrar's office.",
    "official letter kaise milega": "Official letter, jaise bonafide certificate, ke liye student portal ya registrar's office me request submit karein.",
    "आधिकारिक पत्र कैसे प्राप्त करें": "आधिकारिक पत्र, जैसे बोनाफाइड सर्टिफिकेट, के लिए स्टूडेंट पोर्टल या रजिस्ट्रार कार्यालय में आवेदन करें।",
    "अधिकृत पत्र कसे मिळवायचे": "अधिकृत पत्रासाठी, जसे बोनाफाईड सर्टिफिकेट, विद्यार्थी पोर्टलद्वारे किंवा रजिस्ट्रारच्या कार्यालयात अर्ज करा.",

    "what to do if I lost my ID card": "Report the loss to the administration office and apply for a new one by paying the reissuance fee.",
    "agar ID card kho gaya to kya karein": "ID card kho jane par administration office me report karein aur naya banwane ke liye reissuance fee dekar apply karein.",
    "अगर आईडी कार्ड खो गया तो क्या करें": "आईडी कार्ड खो जाने पर प्रशासन कार्यालय में रिपोर्ट करें और नया बनवाने के लिए पुनः जारी करने का शुल्क देकर आवेदन करें।",
    "ओळखपत्र हरवल्यास काय करावे": "ओळखपत्र हरवल्याची तक्रार प्रशासन कार्यालयात करा आणि पुन्हा बनवण्यासाठी शुल्क भरून अर्ज करा.",
    
    # 18 Career Services
    "how to prepare for an interview": "Attend mock interview sessions conducted by the placement cell and review common interview questions.",
    "interview ke liye kaise prepare karein": "Placement cell dwara hone wale mock interview sessions attend karein aur common interview questions padhein.",
    "इंटरव्यू के लिए कैसे तैयारी करें": "प्लेसमेंट सेल द्वारा आयोजित मॉक इंटरव्यू सत्रों में भाग लें और सामान्य इंटरव्यू प्रश्नों की समीक्षा करें।",
    "इंटरव्ह्यूसाठी तयारी कशी करावी": "प्लेसमेंट सेलद्वारे आयोजित मॉक इंटरव्ह्यूमध्ये सहभागी व्हा आणि सामान्य इंटरव्ह्यू प्रश्नांचा अभ्यास करा.",

    "do you have a resume review service": "Yes, the career services department offers resume review and feedback sessions. Book an appointment online.",
    "kya resume review service hai": "Haan, career services department resume review aur feedback sessions offer karta hai. Online appointment book karein.",
    "क्या रिज्यूमे रिव्यू सेवा है": "हाँ, करियर सर्विसेज विभाग रिज्यूमे रिव्यू और फीडबैक सत्र प्रदान करता है। ऑनलाइन अपॉइंटमेंट बुक करें।",
    "रिझ्युमे रिव्ह्यू सेवा उपलब्ध आहे का": "हो, करिअर सेवा विभाग रिझ्युमे रिव्ह्यू आणि फीडबॅक सत्र देतात. ऑनलाइन अपॉइंटमेंट बुक करा.",
    
    "when is the next job fair": "The next job fair is scheduled for [Date]. Check the career services portal for the list of companies attending.",
    "agla job fair kab hai": "Agla job fair [Date] ko hai. Career services portal par aane wali companies ki list check karein.",
    "अगला जॉब फेयर कब है": "अगला जॉब फेयर [तारीख] को निर्धारित है। भाग लेने वाली कंपनियों की सूची के लिए करियर सर्विसेज पोर्टल देखें।",
    "पुढील जॉब फेअर कधी आहे": "पुढील जॉब फेअर [तारीख] ला आहे. सहभागी होणाऱ्या कंपन्यांची यादी करिअर सेवा पोर्टलवर तपासा.",
    
    # 19 Campus Security & Grievance
    "how to report a grievance": "Submit your grievance through the online grievance redressal portal or contact the grievance cell directly via email.",
    "grievance kaise report karein": "Online grievance redressal portal ke through ya email se grievance cell ko direct contact karein.",
    "शिकायत कैसे दर्ज करें": "ऑनलाइन शिकायत निवारण पोर्टल के माध्यम से या सीधे ईमेल द्वारा शिकायत सेल से संपर्क करें।",
    "तक्रार कशी नोंदवायची": "ऑनलाइन तक्रार निवारण पोर्टलद्वारे किंवा थेट ईमेलद्वारे तक्रार निवारण कक्षाशी संपर्क साधा.",

    "what is the anti-ragging policy": "Ragging is strictly prohibited. Report any incidents to the anti-ragging committee or a faculty member immediately.",
    "anti-ragging policy kya hai": "Ragging bilkul mana hai. Koi bhi incident anti-ragging committee ya kisi faculty member ko turant report karein.",
    "एंटी-रैगिंग नीति क्या है": "रैगिंग सख्त वर्जित है। किसी भी घटना की रिपोर्ट तुरंत एंटी-रैगिंग समिति या किसी संकाय सदस्य को करें।",
    "अँटी-रॅगिंग धोरण काय आहे": "रॅगिंग पूर्णपणे प्रतिबंधित आहे. कोणत्याही घटनेची तक्रार त्वरित अँटी-रॅगिंग समिती किंवा प्राध्यापकाकडे करा.",
    
    "where is the lost and found office": "The lost and found office is located at [location]. You can submit a report there or check for lost items.",
    "lost and found office kahan hai": "Lost and found office [location] par hai. Wahan aap report submit kar sakte hain ya khoi hui cheezein check kar sakte hain.",
    "खोया-पाया कार्यालय कहाँ है": "खोया-पाया कार्यालय [स्थान] पर स्थित है। आप वहाँ रिपोर्ट जमा कर सकते हैं या खोई हुई वस्तुओं की जाँच कर सकते हैं।",
    "हरवलेली वस्तू आणि सापडलेली वस्तूंचे कार्यालय कुठे आहे": "हरवलेली वस्तू आणि सापडलेली वस्तूंचे कार्यालय [स्थान] येथे आहे. तुम्ही तिथे तक्रार दाखल करू शकता किंवा हरवलेल्या वस्तू तपासू शकता.",

    # 20 Academic Support & Tutoring
    "are tutoring services available": "Yes, free tutoring is available for various subjects. Check the academic support center's website for schedule and booking.",
    "kya tutoring services hain": "Haan, free tutoring kai subjects ke liye available hai. Academic support center ki website par schedule aur booking check karein.",
    "क्या ट्यूशन सेवाएं उपलब्ध हैं": "हाँ, विभिन्न विषयों के लिए मुफ्त ट्यूशन उपलब्ध है। शेड्यूल और बुकिंग के लिए अकादमिक सहायता केंद्र की वेबसाइट देखें।",
    "शिक्षण सेवा उपलब्ध आहेत का": "हो, विविध विषयांसाठी मोफत शिकवणी उपलब्ध आहे. शेड्यूल आणि बुकिंगसाठी शैक्षणिक समर्थन केंद्राची वेबसाइट तपासा.",

    "how to get help with assignments": "Contact your course professor during office hours or visit the academic writing center for assistance with assignments.",
    "assignments me help kaise milegi": "Assignments me help ke liye course professor se office hours me milein ya academic writing center jayen.",
    "असाइनमेंट में मदद कैसे मिलेगी": "असाइनमेंट में मदद के लिए अपने कोर्स प्रोफेसर से ऑफिस आवर्स में मिलें या शैक्षणिक लेखन केंद्र जाएँ।",
    "असाइनमेंटमध्ये मदत कशी मिळेल": "असाइनमेंटमध्ये मदतीसाठी कोर्स प्राध्यापकांना ऑफिसच्या वेळेत भेटा किंवा शैक्षणिक लेखन केंद्रात जा.",

    "what resources are there for students with disabilities": "The disability services office provides academic accommodations and support. Contact them to discuss your needs.",
    "disability wale students ke liye kya resources hain": "Disability services office academic accommodations aur support deta hai. Apni needs discuss karne ke liye unhe contact karein.",
    "दिव्यांग छात्रों के लिए क्या संसाधन हैं": "दिव्यांग सेवा कार्यालय शैक्षणिक आवास और सहायता प्रदान करता है। अपनी आवश्यकताओं पर चर्चा करने के लिए उनसे संपर्क करें।",
    "अपंग विद्यार्थ्यांसाठी काय संसाधने आहेत": "अपंग सेवा कार्यालय शैक्षणिक सुविधा आणि समर्थन पुरवते. तुमच्या गरजांबद्दल चर्चा करण्यासाठी त्यांच्याशी संपर्क साधा.",

    # 21 Extracurricular Activities
    "how to start a new club": "Propose a new club to the student affairs office. You will need to submit a proposal with a faculty advisor and a list of members.",
    "naya club kaise shuru karein": "Naye club ke liye student affairs office me proposal dein. Aapko faculty advisor aur members ki list ke saath proposal submit karna hoga.",
    "नया क्लब कैसे शुरू करें": "नया क्लब शुरू करने के लिए स्टूडेंट अफेयर्स ऑफिस में एक प्रस्ताव दें। आपको एक संकाय सलाहकार और सदस्यों की सूची के साथ प्रस्ताव प्रस्तुत करना होगा।",
    "नवीन क्लब कसा सुरू करावा": "नवीन क्लब सुरू करण्यासाठी विद्यार्थी व्यवहार कार्यालयाकडे प्रस्ताव द्या. तुम्हाला प्राध्यापक मार्गदर्शक आणि सदस्यांच्या यादीसह प्रस्ताव सादर करावा लागेल.",

    "are there any arts and culture clubs": "Yes, there are several arts and culture clubs. Check the list of student organizations on the college website for details.",
    "kya arts aur culture clubs hain": "Haan, arts aur culture ke kai clubs hain. Details ke liye college website par student organizations ki list check karein.",
    "क्या कला और संस्कृति क्लब हैं": "हाँ, कई कला और संस्कृति क्लब हैं। विवरण के लिए कॉलेज की वेबसाइट पर छात्र संगठनों की सूची देखें।",
    "कला आणि संस्कृतीचे क्लब आहेत का": "हो, अनेक कला आणि संस्कृती क्लब आहेत. तपशिलांसाठी कॉलेज वेबसाइटवरील विद्यार्थी संघटनांची यादी तपासा.",

    "how to participate in inter-college competitions": "Contact the sports or cultural committee to register for inter-college competitions. They handle the nominations and logistics.",
    "inter-college competitions me kaise participate karein": "Inter-college competitions me participate karne ke liye sports ya cultural committee ko contact karein. Wo nominations aur logistics manage karte hain.",
    "इंटरकॅलेज प्रतियोगिताओं में कैसे भाग लें": "इंटरकॅलेज प्रतियोगिताओं में भाग लेने के लिए स्पोर्ट्स या सांस्कृतिक समिति से संपर्क करें। वे नामांकन और लॉजिस्टिक्स संभालते हैं।",
    "आंतर-कॉलेज स्पर्धांमध्ये कसे भाग घ्यावे": "आंतर-कॉलेज स्पर्धांमध्ये भाग घेण्यासाठी क्रीडा किंवा सांस्कृतिक समितीशी संपर्क साधा. ते नामांकन आणि व्यवस्थापन करतात.",

    # 22 Transportation & Parking
    "is parking available on campus": "Yes, student parking is available on campus. You must register your vehicle with the security office to get a parking permit.",
    "kya campus par parking hai": "Haan, campus par student parking available hai. Parking permit lene ke liye aapko security office me apni gadi register karni hogi.",
    "क्या कैंपस पर पार्किंग उपलब्ध है": "हाँ, कैंपस पर छात्रों के लिए पार्किंग उपलब्ध है। आपको पार्किंग परमिट प्राप्त करने के लिए सुरक्षा कार्यालय में अपना वाहन पंजीकृत कराना होगा।",
    "कॅम्पसवर पार्किंग उपलब्ध आहे का": "हो, कॅम्पसवर विद्यार्थ्यांसाठी पार्किंग उपलब्ध आहे. पार्किंग परमिट मिळवण्यासाठी तुम्हाला सुरक्षा कार्यालयात तुमचे वाहन नोंदणी करावे लागेल.",

    "are there shuttle services": "Yes, the college provides shuttle services to and from major locations. Check the transport office for routes and timings.",
    "kya shuttle services hain": "Haan, college major locations tak shuttle services provide karta hai. Routes aur timings ke liye transport office se check karein.",
    "क्या शटल सेवाएं हैं": "हाँ, कॉलेज प्रमुख स्थानों तक शटल सेवाएं प्रदान करता है। मार्गों और समय के लिए परिवहन कार्यालय से जाँच करें।",
    "शटल सेवा उपलब्ध आहे का": "हो, कॉलेज प्रमुख ठिकाणांसाठी शटल सेवा पुरवते. मार्ग आणि वेळेसाठी परिवहन कार्यालयाशी संपर्क साधा.",

    # 23 Student Financial Services & Loans
    "how to get an education loan": "The college has tie-ups with banks for education loans. Contact the financial aid office for a list of partner banks and required documents.",
    "education loan kaise milega": "College ne banks ke saath tie-ups kiye hain education loans ke liye. Partner banks aur documents ki list ke liye financial aid office se contact karein.",
    "एजुकेशन लोन कैसे मिलेगा": "कॉलेज ने एजुकेशन लोन के लिए बैंकों के साथ टाई-अप किया है। पार्टनर बैंकों और आवश्यक दस्तावेजों की सूची के लिए वित्तीय सहायता कार्यालय से संपर्क करें।",
    "शिक्षण कर्ज कसे मिळेल": "कॉलेजने शिक्षण कर्जासाठी बँकांसोबत करार केले आहेत. भागीदार बँका आणि आवश्यक कागदपत्रांच्या यादीसाठी वित्तीय सहाय्य कार्यालयाशी संपर्क साधा.",

    "what documents are needed for a loan": "Generally, you need: admission letter, fee structure, academic records, and proof of income for the guarantor.",
    "loan ke liye kaunse documents chahiye": "Aam taur par, aapko chahiye: admission letter, fee structure, academic records, aur guarantor ke liye income proof.",
    "लोन के लिए कौन से दस्तावेज़ चाहिए": "सामान्यतः, आपको चाहिए: प्रवेश पत्र, शुल्क संरचना, शैक्षणिक रिकॉर्ड, और गारंटर के लिए आय का प्रमाण।",
    "कर्जासाठी कोणती कागदपत्रे लागतात": "सामान्यतः, तुम्हाला हवे आहेत: प्रवेश पत्र, शुल्क रचना, शैक्षणिक रेकॉर्ड आणि गॅरंटरसाठी उत्पन्नाचा पुरावा.",

    # 24 Research & Innovation
    "how to apply for a research grant": "Contact the research and development cell. They provide information on available grants and assist with the application process.",
    "research grant ke liye kaise apply karein": "Research and development cell se contact karein. Wo available grants ki information dete hain aur application process me help karte hain.",
    "रिसर्च ग्रांट के लिए कैसे आवेदन करें": "रिसर्च अँड डेव्हलपमेंट सेल से संपर्क करें। वे उपलब्ध ग्रांट्स की जानकारी देते हैं और आवेदन प्रक्रिया में मदद करते हैं।",
    "संशोधन अनुदानसाठी कसा अर्ज करावा": "संशोधन आणि विकास कक्षाशी संपर्क साधा. ते उपलब्ध अनुदानांची माहिती देतात आणि अर्ज प्रक्रियेत मदत करतात.",

    "is there an incubation center on campus": "Yes, our college has an incubation center to support student startups and innovative projects. Contact the center head for more information.",
    "kya campus par incubation center hai": "Haan, hamare college me incubation center hai student startups aur innovative projects ko support karne ke liye. Zyada information ke liye center head se contact karein.",
    "क्या कैंपस पर इन्क्यूबेशन सेंटर है": "हाँ, हमारे कॉलेज में छात्र स्टार्टअप्स और इनोवेटिव प्रोजेक्ट्स का समर्थन करने के लिए एक इन्क्यूबेशन सेंटर है। अधिक जानकारी के लिए सेंटर हेड से संपर्क करें।",
    "कॅम्पसवर इनक्युबेशन सेंटर आहे का": "हो, आमच्या कॉलेजमध्ये विद्यार्थ्यांच्या स्टार्टअप आणि नवीन प्रोजेक्ट्सला मदत करण्यासाठी इनक्युबेशन सेंटर आहे. अधिक माहितीसाठी सेंटर हेडशी संपर्क साधा.",

    # 25 Mentorship & Advising
    "how to choose a faculty advisor": "Faculty advisors are assigned by the department. You can meet them during office hours to discuss your academic and career goals.",
    "faculty advisor kaise choose karein": "Faculty advisors department dwara assign hote hain. Aap unse office hours me milkar academic aur career goals discuss kar sakte hain.",
    "फैकल्टी एडवाइजर कैसे चुनें": "फैकल्टी एडवाइजर्स विभाग द्वारा नियुक्त किए जाते हैं। आप उनसे ऑफिस आवर्स में मिलकर अपने शैक्षणिक और करियर लक्ष्यों पर चर्चा कर सकते हैं।",
    "प्राध्यापक सल्लागार कसा निवडायचा": "प्राध्यापक सल्लागार विभागाद्वारे नियुक्त केले जातात. तुम्ही त्यांना ऑफिसच्या वेळेत भेटून तुमच्या शैक्षणिक आणि करिअरच्या उद्दिष्टांवर चर्चा करू शकता.",

    "is there a peer mentorship program": "Yes, we have a peer mentorship program where senior students guide new students. Contact the student welfare office for details and sign-up.",
    "kya peer mentorship program hai": "Haan, hamare yahan peer mentorship program hai jahan senior students naye students ko guide karte hain. Details aur sign-up ke liye student welfare office se contact karein.",
    "क्या पीयर मेंटरशिप प्रोग्राम है": "हाँ, हमारे यहाँ एक पीयर मेंटरशिप प्रोग्राम है जहाँ वरिष्ठ छात्र नए छात्रों को मार्गदर्शन देते हैं। विवरण और साइन-अप के लिए छात्र कल्याण कार्यालय से संपर्क करें।",
    "पीअर मेंटरशिप कार्यक्रम आहे का": "हो, आमच्याकडे पीअर मेंटरशिप कार्यक्रम आहे जिथे वरिष्ठ विद्यार्थी नवीन विद्यार्थ्यांना मार्गदर्शन करतात. तपशील आणि नावनोंदणीसाठी विद्यार्थी कल्याण कार्यालयाशी संपर्क साधा.",

    # 26 Feedback & Grievance Redressal
    "how to give feedback on a course": "You can provide course feedback through the online survey conducted at the end of each semester or directly contact your department head.",
    "course par feedback kaise dein": "Aap har semester ke ant me hone wale online survey ke through ya direct apne department head ko contact karke feedback de sakte hain.",
    "कोर्स पर फीडबैक कैसे दें": "आप प्रत्येक सेमेस्टर के अंत में आयोजित ऑनलाइन सर्वे के माध्यम से या सीधे अपने विभाग प्रमुख से संपर्क करके कोर्स पर फीडबैक दे सकते हैं।",
    "कोर्सवर अभिप्राय कसा द्यावा": "प्रत्येक सेमेस्टरच्या शेवटी आयोजित ऑनलाइन सर्वेद्वारे किंवा थेट तुमच्या विभागप्रमुखांशी संपर्क साधून तुम्ही कोर्सवर अभिप्राय देऊ शकता.",
    
    "what to do if i have a complaint": "Submit your complaint through the official grievance redressal portal. All complaints are handled confidentially.",
    "agar koi complaint ho to kya karein": "Apni complaint official grievance redressal portal ke through submit karein. Sabhi complaints ko confidential rakha jata hai.",
    "अगर कोई शिकायत हो तो क्या करें": "अपनी शिकायत आधिकारिक शिकायत निवारण पोर्टल के माध्यम से सबमिट करें। सभी शिकायतों को गोपनीय रखा जाता है।",
    "जर काही तक्रार असेल तर काय करावे": "तुमची तक्रार अधिकृत तक्रार निवारण पोर्टलद्वारे सबमिट करा. सर्व तक्रारी गोपनीय ठेवल्या जातात.",

    # 27 International Student Support
    "what services are available for international students": "The international student office provides assistance with visa processes, cultural orientation, and accommodation services.",
    "international students ke liye kya services hain": "International student office visa process, cultural orientation, aur accommodation services me help karta hai.",
    "अंतरराष्ट्रीय छात्रों के लिए क्या सेवाएं उपलब्ध हैं": "अंतरराष्ट्रीय छात्र कार्यालय वीजा प्रक्रिया, सांस्कृतिक मार्गदर्शन और आवास सेवाओं में सहायता प्रदान करता है।",
    "आंतरराष्ट्रीय विद्यार्थ्यांसाठी कोणत्या सेवा उपलब्ध आहेत": "आंतरराष्ट्रीय विद्यार्थी कार्यालय व्हिसा प्रक्रिया, सांस्कृतिक मार्गदर्शन आणि निवास सेवांमध्ये मदत करते.",

     # 28 Campus Events & Workshops
    "how to register for a workshop": "Registration for workshops is usually done through the student portal or a specific event registration link shared by the organizing department.",
    "workshop ke liye kaise register karein": "Workshop ke liye registration aam taur par student portal ya organizing department dwara share kiye gaye event registration link se hota hai.",
    "वर्कशॉप के लिए कैसे रजिस्टर करें": "वर्कशॉप के लिए पंजीकरण आमतौर पर छात्र पोर्टल या आयोजक विभाग द्वारा साझा किए गए एक विशिष्ट इवेंट पंजीकरण लिंक के माध्यम से होता है।",
    "कार्यशाळेसाठी नोंदणी कशी करावी": "कार्यशाळेसाठी नोंदणी सहसा विद्यार्थी पोर्टल किंवा आयोजक विभागाने शेअर केलेल्या विशिष्ट कार्यक्रम नोंदणी लिंकद्वारे केली जाते.",

    "where can i find information about cultural events": "Details about cultural events are posted on the college website's events calendar, social media channels, and the main notice boards.",
    "cultural events ki jankari kahan milegi": "Cultural events ki jankari college ki website, social media channels aur main notice boards par mil jayegi.",
    "सांस्कृतिक कार्यक्रमों की जानकारी कहाँ मिलेगी": "सांस्कृतिक कार्यक्रमों के बारे में जानकारी कॉलेज की वेबसाइट के इवेंट्स कैलेंडर, सोशल मीडिया चैनलों और मुख्य नोटिस बोर्ड पर पोस्ट की जाती है।",
    "सांस्कृतिक कार्यक्रमांची माहिती कुठे मिळेल": "सांस्कृतिक कार्यक्रमांबद्दलची माहिती कॉलेजच्या वेबसाइटवरील इव्हेंट कॅलेंडर, सोशल मीडिया चॅनेल्स आणि मुख्य नोटीस बोर्डवर पोस्ट केली जाते.",

    # 29 IT Help & Software
    "how to get access to college software": "College software and licenses are managed by the IT department. You can log in with your college credentials to access them.",
    "college software kaise access karein": "College ke software aur licenses IT department manage karta hai. Aap unhe apne college credentials se login karke access kar sakte hain.",
    "कॉलेज सॉफ्टवेयर कैसे एक्सेस करें": "कॉलेज सॉफ्टवेयर और लाइसेंस IT विभाग द्वारा प्रबंधित किए जाते हैं। आप उन्हें अपने कॉलेज क्रेडेंशियल के साथ लॉग इन करके एक्सेस कर सकते हैं।",
    "कॉलेज सॉफ्टवेअर कसे ऍक्सेस करावे": "कॉलेजचे सॉफ्टवेअर आणि लायसन्स IT विभागाद्वारे व्यवस्थापित केले जातात. तुम्ही तुमच्या कॉलेजच्या क्रेडेन्शियल्सनी लॉग इन करून त्यांना ऍक्सेस करू शकता.",

    "my student account is locked, what do i do": "If your student account is locked, contact the IT helpdesk immediately via phone or email for assistance with unlocking it.",
    "mera student account lock ho gaya, kya karoon": "Agar aapka student account lock ho gaya hai, to turant IT helpdesk ko phone ya email se contact karein madad ke liye.",
    "मेरा छात्र खाता लॉक हो गया है, मैं क्या करूँ": "यदि आपका छात्र खाता लॉक हो गया है, तो इसे अनलॉक करने में सहायता के लिए तुरंत आईटी हेल्पडेस्क से फोन या ईमेल द्वारा संपर्क करें।",
    "माझे विद्यार्थी खाते लॉक झाले आहे, मी काय करू": "तुमचे विद्यार्थी खाते लॉक झाले असल्यास, ते अनलॉक करण्यासाठी मदत मिळवण्यासाठी ताबडतोब IT हेल्पडेस्कशी फोन किंवा ईमेलद्वारे संपर्क साधा.",

    # 30 Campus Infrastructure & Services
    "where can i find a campus map": "A campus map is available on the college website under the 'Campus' section. You can also find physical maps at the main gates and administrative buildings.",
    "campus map kahan milega": "Campus map college ki website ke 'Campus' section me available hai. Aapko physical maps main gate aur administrative buildings par bhi mil sakte hain.",
    "कैंपस का नक्शा कहाँ मिलेगा": "कैंपस का नक्शा कॉलेज की वेबसाइट पर 'कैंपस' अनुभाग के तहत उपलब्ध है। आप मुख्य द्वारों और प्रशासनिक भवनों पर भी भौतिक नक्शे पा सकते हैं।",
    "कॅम्पसचा नकाशा कुठे मिळेल": "कॅम्पसचा नकाशा कॉलेजच्या वेबसाइटवरील 'कॅम्पस' विभागात उपलब्ध आहे. तुम्हाला मुख्य गेट्स आणि प्रशासकीय इमारतींवर देखील भौतिक नकाशे मिळू शकतात.",

    "how to use the campus printing services": "To use printing services, you need to load credit on your student ID card or use the online portal to pay for printouts. Instructions are available at the printing stations.",
    "campus printing services kaise use karein": "Printing services use karne ke liye, aapko student ID card me credit load karna padega ya online portal se printouts ke liye pay karna hoga. Instructions printing stations par available hain.",
    "कैंपस प्रिंटिंग सेवाओं का उपयोग कैसे करें": "प्रिंटिंग सेवाओं का उपयोग करने के लिए, आपको अपने छात्र आईडी कार्ड पर क्रेडिट लोड करना होगा या प्रिंटआउट के लिए भुगतान करने हेतु ऑनलाइन पोर्टल का उपयोग करना होगा। निर्देश प्रिंटिंग स्टेशनों पर उपलब्ध हैं।",
    "कॅम्पस प्रिंटिंग सेवा कशा वापराव्या": "प्रिंटिंग सेवा वापरण्यासाठी, तुम्हाला तुमच्या विद्यार्थी ओळखपत्रावर क्रेडिट लोड करावे लागेल किंवा प्रिंटआउट्ससाठी पैसे देण्यासाठी ऑनलाइन पोर्टल वापरावे लागेल. सूचना प्रिंटिंग स्टेशनवर उपलब्ध आहेत.",

    # 31 Career Development & Counseling
    "how can i get career counseling": "The career services department offers one-on-one counseling sessions to help students with their career path. You can book an appointment through their portal.",
    "career counseling kaise milegi": "Career services department students ko unke career path me help karne ke liye one-on-one counseling sessions offer karta hai. Unke portal se appointment book karein.",
    "करियर काउंसलिंग कैसे मिलेगी": "करियर सर्विसेज विभाग छात्रों को उनके करियर पथ में मदद करने के लिए वन-ऑन-वन काउंसलिंग सत्र प्रदान करता है। आप उनके पोर्टल के माध्यम से अपॉइंटमेंट बुक कर सकते हैं।",
    "करिअर समुपदेशन कसे मिळवू": "करिअर सेवा विभाग विद्यार्थ्यांना त्यांच्या करिअर मार्गात मदत करण्यासाठी वन-ऑन-वन समुपदेशन सत्रे देतात. तुम्ही त्यांच्या पोर्टलद्वारे अपॉइंटमेंट बुक करू शकता.",

    # 32 Student Union & Governance
    "how to run for student council elections": "To run for a position, you must fill out the nomination form from the Student Affairs office and submit it by the announced deadline.",
    "student council elections me kaise hissa lein": "Position ke liye hissa lene ke liye, aapko Student Affairs office se nomination form bharna hoga aur usse announce kiye gaye deadline tak submit karna hoga.",
    "छात्र परिषद चुनाव में कैसे भाग लें": "पद के लिए चुनाव लड़ने के लिए, आपको छात्र मामलों के कार्यालय से नामांकन फॉर्म भरना होगा और उसे घोषित समय सीमा तक जमा करना होगा।",
    "विद्यार्थी परिषद निवडणुकीत कसे उभे राहावे": "पदासाठी निवडणूक लढवण्यासाठी, तुम्हाला विद्यार्थी व्यवहार कार्यालयातून नामांकन फॉर्म भरावा लागेल आणि घोषित केलेल्या मुदतीपर्यंत तो जमा करावा लागेल.",

    "what are the roles of the student council members": "Student council members represent student interests, organize events, and act as a bridge between the students and the administration.",
    "student council members ke roles kya hain": "Student council members students ke interests ko represent karte hain, events organize karte hain aur students aur administration ke beech ek bridge ka kaam karte hain.",
    "छात्र परिषद सदस्यों की भूमिकाएँ क्या हैं": "छात्र परिषद सदस्य छात्रों के हितों का प्रतिनिधित्व करते हैं, कार्यक्रमों का आयोजन करते हैं, और छात्रों और प्रशासन के बीच एक सेतु का काम करते हैं।",
    "विद्यार्थी परिषद सदस्यांची भूमिका काय आहे": "विद्यार्थी परिषद सदस्य विद्यार्थ्यांच्या हिताचे प्रतिनिधित्व करतात, कार्यक्रम आयोजित करतात आणि विद्यार्थी आणि प्रशासनामध्ये दुवा म्हणून काम करतात.",

    # 33 College Newsletter & Media
    "how to submit an article to the college newsletter": "You can submit articles to the college newsletter by emailing the editor. Check the newsletter's website for submission guidelines and deadlines.",
    "college newsletter me article kaise submit karein": "College newsletter me articles submit karne ke liye aap editor ko email kar sakte hain. Submission guidelines aur deadlines ke liye newsletter ki website check karein.",
    "कॉलेज न्यूज़लेटर में लेख कैसे जमा करें": "आप संपादक को ईमेल करके कॉलेज न्यूज़लेटर में लेख जमा कर सकते हैं। जमा करने के दिशा-निर्देशों और समय-सीमाओं के लिए न्यूज़लेटर की वेबसाइट देखें।",
    "कॉलेज वृत्तपत्रिकेमध्ये लेख कसा सादर करावा": "तुम्ही संपादकाला ईमेल करून कॉलेज वृत्तपत्रिकेत लेख सादर करू शकता. सबमिशन मार्गदर्शक तत्त्वे आणि मुदतीसाठी वृत्तपत्रिकेची वेबसाइट तपासा.",

    # 34 Study Abroad & Exchange Programs
    "what are the eligibility criteria for exchange programs": "Eligibility for exchange programs usually includes a minimum GPA, language proficiency, and a clear academic record. Consult the international relations office for details.",
    "exchange programs ke liye eligibility kya hai": "Exchange programs ke liye eligibility me aam taur par ek minimum GPA, language proficiency aur ek clear academic record shamil hota hai. Details ke liye international relations office se consult karein.",
    "एक्सचेंज प्रोग्राम के लिए पात्रता मानदंड क्या हैं": "एक्सचेंज प्रोग्राम के लिए पात्रता में आमतौर पर एक न्यूनतम जीपीए, भाषा दक्षता और एक स्पष्ट शैक्षणिक रिकॉर्ड शामिल होता है। विवरण के लिए अंतर्राष्ट्रीय संबंध कार्यालय से परामर्श करें।",
    "एक्सचेंज कार्यक्रमासाठी पात्रता निकष काय आहेत": "एक्सचेंज कार्यक्रमासाठी पात्रतेमध्ये सामान्यतः किमान जीपीए, भाषिक प्राविण्य आणि स्पष्ट शैक्षणिक रेकॉर्ड समाविष्ट असते. तपशिलांसाठी आंतरराष्ट्रीय संबंध कार्यालयाचा सल्ला घ्या.",

    "how to apply for a study abroad program": "The application process involves submitting an application form, academic transcripts, and a statement of purpose. The study abroad office can guide you through the steps.",
    "study abroad program ke liye kaise apply karein": "Application process me application form, academic transcripts aur statement of purpose submit karna hota hai. Study abroad office aapko steps me guide kar sakta hai.",
    "स्टडी अब्रॉड प्रोग्राम के लिए कैसे आवेदन करें": "आवेदन प्रक्रिया में एक आवेदन फॉर्म, शैक्षणिक ट्रांसक्रिप्ट और एक उद्देश्य का विवरण जमा करना शामिल है। स्टडी अब्रॉड कार्यालय आपको चरणों के माध्यम से मार्गदर्शन कर सकता है।",
    "परदेशातील अभ्यासक्रमासाठी कसा अर्ज करावा": "अर्ज प्रक्रियेमध्ये अर्ज फॉर्म, शैक्षणिक ट्रान्सक्रिप्ट आणि उद्देशाचे विवरण सादर करणे समाविष्ट आहे. परदेशी अभ्यास कार्यालय तुम्हाला या टप्प्यांमध्ये मार्गदर्शन करू शकते.",

    # 35 Extracurricular & Skill Development
    "are there any robotics or coding clubs": "Yes, the college has a dedicated robotics club and a coding society. You can find their contact information and meeting schedules on the student activities page.",
    "kya robotics ya coding clubs hain": "Haan, college me ek dedicated robotics club aur coding society hai. Aap unki contact information aur meeting schedules student activities page par dhoond sakte hain.",
    "क्या रोबोटिक्स या कोडिंग क्लब हैं": "हाँ, कॉलेज में एक समर्पित रोबोटिक्स क्लब और एक कोडिंग सोसाइटी है। आप उनकी संपर्क जानकारी और मीटिंग शेड्यूल छात्र गतिविधियों के पेज पर पा सकते हैं।",
    "रोबोटिक्स किंवा कोडिंग क्लब आहेत का": "हो, कॉलेजमध्ये एक समर्पित रोबोटिक्स क्लब आणि कोडिंग सोसायटी आहे. तुम्ही त्यांची संपर्क माहिती आणि मीटिंगचे वेळापत्रक विद्यार्थी क्रियाकलाप पानावर शोधू शकता.",
    
    # 36 Safety & Emergency
    "what are the fire safety procedures on campus": "In case of a fire, activate the nearest fire alarm, evacuate the building immediately through the designated exits, and assemble at the marked assembly point.",
    "campus par aag lagne par kya karein": "Aag lagne par, sabse paas wale fire alarm ko activate karein, turant designated exits se building khali karein, aur assembly point par ikattha ho jayen.",
    "कैंपस पर आग लगने पर क्या करें": "आग लगने की स्थिति में, निकटतम फायर अलार्म को सक्रिय करें, निर्धारित निकास द्वारों से तुरंत इमारत खाली करें, और चिह्नित सभा स्थल पर इकट्ठा हों।",
    "कॅम्पसवर आग लागल्यास काय करावे": "आग लागल्यास, सर्वात जवळचा फायर अलार्म चालू करा, निर्धारित बाहेर पडण्याच्या मार्गातून त्वरित इमारत रिकामी करा आणि निश्चित केलेल्या ठिकाणी जमा व्हा.",

    "is there a first aid kit available": "Yes, first aid kits are available in every department, laboratory, and the campus medical center. Contact a staff member or faculty for assistance.",
    "kya first aid kit available hai": "Haan, first aid kit har department, laboratory aur campus medical center me available hai. Madad ke liye kisi staff member ya faculty se contact karein.",
    "क्या फर्स्ट एड किट उपलब्ध है": "हाँ, फर्स्ट एड किट हर विभाग, प्रयोगशाला और कैंपस मेडिकल सेंटर में उपलब्ध हैं। सहायता के लिए किसी स्टाफ सदस्य या फैकल्टी से संपर्क करें।",
    "फर्स्ट एड किट उपलब्ध आहे का": "हो, फर्स्ट एड किट प्रत्येक विभाग, प्रयोगशाळा आणि कॅम्पस वैद्यकीय केंद्रात उपलब्ध आहे. मदतीसाठी कर्मचारी किंवा प्राध्यापकाशी संपर्क साधा.",

    # 37 Academic Integrity & Plagiarism
    "what is the college's policy on plagiarism": "Plagiarism is a serious academic offense. The policy requires all submitted work to be original and properly cited. Violations may result in disciplinary action.",
    "plagiarism par college ki policy kya hai": "Plagiarism ek serious academic offense hai. Policy ke mutabik sabhi submitted work original hona chahiye aur usme proper citation hona chahiye. Iska ullaṅghan karne par disciplinary action ho sakta hai.",
    "साहित्यिक चोरी पर कॉलेज की नीति क्या है": "साहित्यिक चोरी एक गंभीर शैक्षणिक अपराध है। नीति के अनुसार सभी सबमिट किए गए काम मूल और ठीक से उद्धृत होने चाहिए। उल्लंघन पर अनुशासनात्मक कार्रवाई हो सकती है।",
    "साहित्यिक चोरीवर कॉलेजचे धोरण काय आहे": "साहित्यिक चोरी हा एक गंभीर शैक्षणिक गुन्हा आहे. धोरणानुसार सर्व सादर केलेले काम मूळ आणि योग्यरित्या उद्धृत केलेले असावे. नियमांचे उल्लंघन केल्यास शिस्तभंगाची कारवाई होऊ शकते.",

    "are there tools to check for plagiarism": "Yes, the college provides access to plagiarism checking software like Turnitin. You can access it through the LMS or the library portal.",
    "kya plagiarism check karne ke tools hain": "Haan, college Turnitin jaise plagiarism checking software ka access provide karta hai. Aap ise LMS ya library portal ke through access kar sakte hain.",
    "क्या साहित्यिक चोरी की जांच के लिए उपकरण हैं": "हाँ, कॉलेज टर्निटिन (Turnitin) जैसे साहित्यिक चोरी की जांच करने वाले सॉफ्टवेयर तक पहुंच प्रदान करता है। आप इसे एलएमएस या लाइब्रेरी पोर्टल के माध्यम से एक्सेस कर सकते हैं।",
    "साहित्यिक चोरी तपासण्यासाठी साधने आहेत का": "हो, कॉलेज टर्निटिन (Turnitin) सारख्या साहित्यिक चोरी तपासणी करणाऱ्या सॉफ्टवेअरची सुविधा देते. तुम्ही ते एलएमएस किंवा लायब्ररी पोर्टलद्वारे ऍक्सेस करू शकता.",

    # 38 Online Learning & Blended Courses
    "how to access online course materials": "Online course materials are available on the Learning Management System (LMS). Log in with your college credentials to view course content, lectures, and assignments.",
    "online course materials kaise access karein": "Online course materials Learning Management System (LMS) par available hain. Course content, lectures aur assignments dekhne ke liye apne college credentials se login karein.",
    "ऑनलाइन पाठ्यक्रम सामग्री कैसे एक्सेस करें": "ऑनलाइन पाठ्यक्रम सामग्री लर्निंग मैनेजमेंट सिस्टम (LMS) पर उपलब्ध है। पाठ्यक्रम सामग्री, व्याख्यान और असाइनमेंट देखने के लिए अपने कॉलेज क्रेडेंशियल के साथ लॉगिन करें।",
    "ऑनलाइन कोर्स साहित्य कसे ऍक्सेस करावे": "ऑनलाइन कोर्स साहित्य लर्निंग मॅनेजमेंट सिस्टीम (LMS) वर उपलब्ध आहे. कोर्समधील सामग्री, व्याख्याने आणि असाइनमेंट्स पाहण्यासाठी तुमच्या कॉलेजच्या क्रेडेन्शियल्सनी लॉग इन करा.",

    "what are the technical requirements for online classes": "You'll need a stable internet connection, a computer or tablet, and a webcam with a microphone for live sessions. Check the IT helpdesk for specific software requirements.",
    "online classes ke liye technical requirements kya hain": "Aapko stable internet connection, computer ya tablet, aur live sessions ke liye webcam aur microphone chahiye. Specific software requirements ke liye IT helpdesk se check karein.",
    "ऑनलाइन कक्षाओं के लिए तकनीकी आवश्यकताएं क्या हैं": "आपको एक स्थिर इंटरनेट कनेक्शन, एक कंप्यूटर या टैबलेट, और लाइव सत्रों के लिए एक वेबकैम और माइक्रोफोन की आवश्यकता होगी। विशिष्ट सॉफ्टवेयर आवश्यकताओं के लिए आईटी हेल्पडेस्क से जांच करें।",
    "ऑनलाइन क्लासेससाठी तांत्रिक आवश्यकता काय आहेत": "तुम्हाला स्थिर इंटरनेट कनेक्शन, एक संगणक किंवा टॅब्लेट, आणि लाइव्ह सत्रांसाठी एक वेबकॅम आणि मायक्रोफोन लागेल. विशिष्ट सॉफ्टवेअर आवश्यकतांसाठी IT हेल्पडेस्कशी संपर्क साधा.",

    # 39 College Services & Facilities
    "where is the photocopy center": "The central photocopy center is located on the ground floor of the library building. You can get photocopies, prints, and stationery items there.",
    "photocopy center kahan hai": "Central photocopy center library building ke ground floor par hai. Wahan se aap photocopies, prints aur stationary items le sakte hain.",
    "फोटोकॉपी सेंटर कहाँ है": "केंद्रीय फोटोकॉपी सेंटर लाइब्रेरी भवन के भूतल पर स्थित है। आप वहाँ से फोटोकॉपी, प्रिंट और स्टेशनरी का सामान ले सकते हैं।",
    "फोटोकॉपी केंद्र कुठे आहे": "केंद्रीय फोटोकॉपी केंद्र ग्रंथालय इमारतीच्या तळमजल्यावर आहे. तेथून तुम्ही फोटोकॉपी, प्रिंट्स आणि स्टेशनरी वस्तू घेऊ शकता.",

    "is there a post office on campus": "Yes, there is a sub-post office on campus located near the main gate. It offers postal services, money transfers, and parcel services.",
    "kya campus par post office hai": "Haan, campus par main gate ke paas ek sub-post office hai. Wo postal services, money transfers aur parcel services offer karta hai.",
    "क्या कैंपस पर पोस्ट ऑफिस है": "हाँ, कैंपस पर मुख्य गेट के पास एक उप-डाकघर है। यह डाक सेवाएँ, मनी ट्रांसफर और पार्सल सेवाएँ प्रदान करता है।",
    "कॅम्पसवर पोस्ट ऑफिस आहे का": "हो, कॅम्पसवर मुख्य गेटजवळ एक उप-टपाल कार्यालय आहे. ते टपाल सेवा, मनी ट्रान्सफर आणि पार्सल सेवा देतात.",

    # 40 Health & Wellness
    "are there mental health services available": "Yes, the student wellness center provides confidential mental health counseling by professional counselors. You can book an appointment through the student portal.",
    "kya mental health services hain": "Haan, student wellness center me professional counselors dwara confidential mental health counseling milti hai. Student portal se appointment book kar sakte hain.",
    "क्या मानसिक स्वास्थ्य सेवाएं उपलब्ध हैं": "हाँ, छात्र कल्याण केंद्र पेशेवर काउंसलर्स द्वारा गोपनीय मानसिक स्वास्थ्य परामर्श प्रदान करता है। आप छात्र पोर्टल के माध्यम से अपॉइंटमेंट बुक कर सकते हैं।",
    "मानसिक आरोग्य सेवा उपलब्ध आहेत का": "हो, विद्यार्थी कल्याण केंद्रामध्ये व्यावसायिक समुपदेशकांद्वारे गोपनीय मानसिक आरोग्य समुपदेशन मिळते. तुम्ही विद्यार्थी पोर्टलद्वारे अपॉइंटमेंट बुक करू शकता.",

    "what kind of medical services are provided at the clinic": "The campus clinic provides first aid, basic medical check-ups, and a limited supply of common medications. For serious conditions, they can provide a referral.",
    "clinic me kaisi medical services milti hain": "Campus clinic me first aid, basic medical check-ups, aur common medications ki limited supply milti hai. Serious conditions ke liye, wo referral de sakte hain.",
    "क्लिनिक में किस तरह की मेडिकल सेवाएं मिलती हैं": "कैंपस क्लिनिक में प्राथमिक उपचार, सामान्य चिकित्सा जांच और सामान्य दवाओं की सीमित आपूर्ति प्रदान की जाती है। गंभीर स्थितियों के लिए, वे रेफरल दे सकते हैं।",
    "क्लिनिकमध्ये कोणत्या प्रकारच्या वैद्यकीय सेवा मिळतात": "कॅम्पस क्लिनिकमध्ये प्रथमोपचार, सामान्य वैद्यकीय तपासणी आणि सामान्य औषधांचा मर्यादित पुरवठा मिळतो. गंभीर परिस्थितीसाठी, ते रेफरल देऊ शकतात.",

    # 41 Academic Records & Verification
    "how to get a duplicate certificate": "To get a duplicate certificate, you need to submit a written application to the records office along with a police report (FIR) and pay the reissuance fee.",
    "duplicate certificate kaise milega": "Duplicate certificate ke liye aapko records office me written application submit karni hogi, saath me police report (FIR) aur reissuance fee bhi deni hogi.",
    "डुप्लिकेट प्रमाणपत्र कैसे मिलेगा": "डुप्लिकेट प्रमाणपत्र प्राप्त करने के लिए, आपको रिकॉर्ड्स कार्यालय में एक लिखित आवेदन, पुलिस रिपोर्ट (एफआईआर) के साथ जमा करना होगा और पुनः जारी करने का शुल्क देना होगा।",
    "डुप्लिकेट प्रमाणपत्र कसे मिळेल": "डुप्लिकेट प्रमाणपत्र मिळवण्यासाठी, तुम्हाला रेकॉर्ड्स ऑफिसमध्ये लिखित अर्ज सादर करावा लागेल, सोबत पोलीस रिपोर्ट (FIR) आणि पुन्हा जारी करण्याचे शुल्क भरावे लागेल.",

    "how long does degree verification take": "Degree verification usually takes 10-15 business days. The process can be expedited by paying an extra fee. Check with the registrar's office for the exact timeline.",
    "degree verification me kitna time lagta hai": "Degree verification me aam taur par 10-15 business days lagte hain. Extra fee dekar process ko expedite kiya ja sakta hai. Exact timeline ke liye registrar's office se check karein.",
    "डिग्री सत्यापन में कितना समय लगता है": "डिग्री सत्यापन में आमतौर पर 10-15 व्यावसायिक दिन लगते हैं। अतिरिक्त शुल्क का भुगतान करके प्रक्रिया को तेज किया जा सकता है। सटीक समय-सीमा के लिए रजिस्ट्रार के कार्यालय से जांच करें।",
    "पदवी पडताळणीला किती वेळ लागतो": "पदवी पडताळणीला साधारणतः 10-15 व्यावसायिक दिवस लागतात. अतिरिक्त शुल्क भरून प्रक्रिया लवकर पूर्ण करता येते. निश्चित वेळेसाठी रजिस्ट्रारच्या कार्यालयाशी संपर्क साधा.",

    # 42 Student Feedback & Grievance Redressal
    "how to file an academic complaint": "To file an academic complaint, submit a detailed written application to the head of your department. You can also use the online grievance portal for a formal record.",
    "academic complaint kaise file karein": "Academic complaint file karne ke liye, apne department head ko ek detailed written application submit karein. Formal record ke liye online grievance portal ka bhi use kar sakte hain.",
    "शैक्षणिक शिकायत कैसे दर्ज करें": "शैक्षणिक शिकायत दर्ज करने के लिए, अपने विभाग प्रमुख को एक विस्तृत लिखित आवेदन जमा करें। औपचारिक रिकॉर्ड के लिए आप ऑनलाइन शिकायत पोर्टल का भी उपयोग कर सकते हैं।",
    "शैक्षणिक तक्रार कशी दाखल करावी": "शैक्षणिक तक्रार दाखल करण्यासाठी, तुमच्या विभागप्रमुखांना एक सविस्तर लेखी अर्ज सादर करा. औपचारिक रेकॉर्डसाठी तुम्ही ऑनलाइन तक्रार पोर्टलचा देखील वापर करू शकता.",

    "is there an ombudsman for student issues": "Yes, an ombudsman is appointed to handle student issues and disputes impartially. You can find their contact information on the college website.",
    "kya students ke liye ombudsman hai": "Haan, students ke issues aur disputes ko unbiased tarike se handle karne ke liye ek ombudsman appoint kiya jata hai. Unki contact information college website par mil jayegi.",
    "क्या छात्रों की समस्याओं के लिए कोई लोकपाल है": "हाँ, छात्रों की समस्याओं और विवादों को निष्पक्ष रूप से संभालने के लिए एक लोकपाल नियुक्त किया गया है। आप उनकी संपर्क जानकारी कॉलेज की वेबसाइट पर पा सकते हैं।",
    "विद्यार्थ्यांच्या समस्यांसाठी लोकपाल आहे का": "हो, विद्यार्थ्यांच्या समस्या आणि वादांचे निष्पक्षपणे निराकरण करण्यासाठी एक लोकपाल नियुक्त केला जातो. त्यांची संपर्क माहिती कॉलेजच्या वेबसाइटवर मिळू शकते.",

    # 43 Campus Clubs & Societies
    "how to get funding for a club event": "To get funding, your club must submit a detailed event proposal and budget to the student affairs office for approval. They will review and allocate funds based on the request.",
    "club event ke liye funding kaise milegi": "Funding ke liye, aapke club ko ek detailed event proposal aur budget Student Affairs office me approval ke liye submit karna hoga. Wo request ke hisab se funds allocate karenge.",
    "क्लब इवेंट के लिए फंडिंग कैसे मिलेगी": "फंडिंग पाने के लिए, आपके क्लब को एक विस्तृत इवेंट प्रस्ताव और बजट छात्र मामलों के कार्यालय में अनुमोदन के लिए जमा करना होगा। वे अनुरोध के आधार पर धन आवंटित करेंगे।",
    "क्लब इव्हेंटसाठी निधी कसा मिळवायचा": "निधी मिळवण्यासाठी, तुमच्या क्लबने एक सविस्तर इव्हेंट प्रस्ताव आणि बजेट विद्यार्थी व्यवहार कार्यालयात मंजुरीसाठी सादर करणे आवश्यक आहे. ते विनंतीनुसार निधीचे वाटप करतील.",

    "how to book a room for a club meeting": "You can book a room for a club meeting through the online room booking portal, available on the college's internal network. You need approval from the student affairs office.",
    "club meeting ke liye room kaise book karein": "Club meeting ke liye room online room booking portal ke through book kar sakte hain, jo college ke internal network par available hai. Iske liye Student Affairs office se approval chahiye.",
    "क्लब मीटिंग के लिए रूम कैसे बुक करें": "क्लब मीटिंग के लिए रूम को ऑनलाइन रूम बुकिंग पोर्टल के माध्यम से बुक किया जा सकता है, जो कॉलेज के आंतरिक नेटवर्क पर उपलब्ध है। इसके लिए छात्र मामलों के कार्यालय से अनुमोदन की आवश्यकता है।",
    "क्लब मीटिंगसाठी रूम कशी बुक करावी": "क्लब मीटिंगसाठी रूम ऑनलाइन रूम बुकिंग पोर्टलद्वारे बुक केली जाऊ शकते, जी कॉलेजच्या अंतर्गत नेटवर्कवर उपलब्ध आहे. यासाठी विद्यार्थी व्यवहार कार्यालयाची मंजुरी आवश्यक आहे.",
    
    # 44 Alumni Network & Engagement
    "how to connect with alumni for mentorship": "You can connect with alumni through the official alumni portal. The career services and alumni relations offices can also help facilitate mentorship connections.",
    "mentorship ke liye alumni se kaise connect karein": "Mentorship ke liye aap official alumni portal ke through alumni se connect kar sakte hain. Career services aur alumni relations offices bhi mentorship connections me help kar sakte hain.",
    "मेंटरशिप के लिए पूर्व छात्रों से कैसे जुड़ें": "मेंटरशिप के लिए आप आधिकारिक पूर्व छात्र पोर्टल के माध्यम से पूर्व छात्रों से जुड़ सकते हैं। करियर सेवा और पूर्व छात्र संबंध कार्यालय भी मेंटरशिप कनेक्शन में मदद कर सकते हैं।",
    "मार्गदर्शनासाठी माजी विद्यार्थ्यांशी कसे जोडले जावे": "मार्गदर्शनासाठी तुम्ही अधिकृत माजी विद्यार्थी पोर्टलद्वारे माजी विद्यार्थ्यांशी कनेक्ट होऊ शकता. करिअर सेवा आणि माजी विद्यार्थी संबंध कार्यालये देखील मार्गदर्शन संबंधांमध्ये मदत करू शकतात.",

    "are there any alumni events I can attend": "Yes, the college organizes various alumni events, including reunions, networking mixers, and guest lectures. Check the alumni portal for upcoming events.",
    "kya koi alumni events hain jo main attend kar sakta hoon": "Haan, college various alumni events organize karta hai, jaise reunions, networking mixers aur guest lectures. Upcoming events ke liye alumni portal check karein.",
    "क्या कोई पूर्व छात्र कार्यक्रम हैं जिनमें मैं भाग ले सकता हूँ": "हाँ, कॉलेज विभिन्न पूर्व छात्र कार्यक्रम आयोजित करता है, जिसमें पुनर्मिलन, नेटवर्किंग मीट और अतिथि व्याख्यान शामिल हैं। आगामी कार्यक्रमों के लिए पूर्व छात्र पोर्टल देखें।",
    "मी उपस्थित राहू शकणारे माजी विद्यार्थ्यांचे कार्यक्रम आहेत का": "हो, कॉलेज विविध माजी विद्यार्थ्यांचे कार्यक्रम आयोजित करते, ज्यात पुनर्मिलन, नेटवर्किंग मिक्सर्स आणि अतिथी व्याख्याने यांचा समावेश आहे. आगामी कार्यक्रमांसाठी माजी विद्यार्थी पोर्टल तपासा.",

    # 45 Sports & Recreation
    "how can I join a college sports team": "To join a college sports team, you must attend the selection trials and register with the sports department. Dates for trials are announced on the college notice board and website.",
    "college sports team kaise join karoon": "College sports team join karne ke liye, aapko selection trials me hissa lena hoga aur sports department me register karna hoga. Trials ki dates college ke notice board aur website par announce hoti hain.",
    "कॉलेज स्पोर्ट्स टीम में कैसे शामिल हों": "कॉलेज स्पोर्ट्स टीम में शामिल होने के लिए, आपको चयन परीक्षणों में भाग लेना होगा और खेल विभाग में पंजीकरण करना होगा। परीक्षणों की तारीखें कॉलेज के नोटिस बोर्ड और वेबसाइट पर घोषित की जाती हैं।",
    "कॉलेज स्पोर्ट्स टीममध्ये कसे सामील व्हावे": "कॉलेज स्पोर्ट्स टीममध्ये सामील होण्यासाठी, तुम्हाला निवड चाचण्यांमध्ये सहभागी व्हावे लागेल आणि क्रीडा विभागात नोंदणी करावी लागेल. चाचण्यांच्या तारखा कॉलेजच्या नोटीस बोर्ड आणि वेबसाइटवर जाहीर केल्या जातात.",

    "are there facilities for individual sports": "Yes, our campus has facilities for individual sports like a swimming pool, a badminton court, and a table tennis room. You can book them through the sports portal.",
    "kya individual sports ke liye facilities hain": "Haan, hamare campus par individual sports jaise swimming pool, badminton court aur table tennis room ke liye facilities hain. Aap unhe sports portal se book kar sakte hain.",
    "क्या व्यक्तिगत खेलों के लिए सुविधाएं हैं": "हाँ, हमारे कैंपस में व्यक्तिगत खेलों के लिए सुविधाएं हैं जैसे एक स्विमिंग पूल, एक बैडमिंटन कोर्ट और एक टेबल टेनिस रूम। आप उन्हें स्पोर्ट्स पोर्टल के माध्यम से बुक कर सकते हैं।",
    "वैयक्तिक खेळांसाठी सुविधा आहेत का": "हो, आमच्या कॅम्पसमध्ये वैयक्तिक खेळांसाठी सुविधा आहेत, जसे की स्विमिंग पूल, बॅडमिंटन कोर्ट आणि टेबल टेनिस रूम. तुम्ही त्यांना स्पोर्ट्स पोर्टलद्वारे बुक करू शकता.",

    # 46 Intellectual Property & Patents
    "how can students get their research patented": "The college's research and innovation cell assists students with the patent filing process. You should first disclose your invention to the cell, who will guide you on the next steps.",
    "students apna research patent kaise karwa sakte hain": "College ka research aur innovation cell students ko patent filing process me assist karta hai. Aapko sabse pehle apni invention cell ko disclose karni hogi, jo aage ke steps me guide karenge.",
    "छात्र अपने शोध का पेटेंट कैसे करवा सकते हैं": "कॉलेज का शोध और नवाचार सेल पेटेंट दाखिल करने की प्रक्रिया में छात्रों की सहायता करता है। आपको सबसे पहले अपने आविष्कार का खुलासा सेल को करना चाहिए, जो आपको अगले चरणों पर मार्गदर्शन करेगा।",
    "विद्यार्थी त्यांच्या संशोधनाचे पेटंट कसे मिळवू शकतात": "कॉलेजचा संशोधन आणि नवोपक्रम कक्ष विद्यार्थ्यांना पेटंट दाखल करण्याच्या प्रक्रियेत मदत करतो. तुम्ही प्रथम तुमचा शोध कक्षाला सांगावा, जो तुम्हाला पुढील चरणांबद्दल मार्गदर्शन करेल.",

    # 47 Community & Social Responsibility
    "what are the volunteering opportunities on campus": "The NSS (National Service Scheme) and various social clubs offer volunteering opportunities for community service. You can register with them to participate in their events and drives.",
    "campus par volunteering opportunities kya hain": "NSS (National Service Scheme) aur various social clubs community service ke liye volunteering opportunities dete hain. Aap unke events aur drives me participate karne ke liye unke saath register kar sakte hain.",
    "कैंपस पर स्वयंसेवा के अवसर क्या हैं": "एनएसएस (राष्ट्रीय सेवा योजना) और विभिन्न सामाजिक क्लब सामुदायिक सेवा के लिए स्वयंसेवा के अवसर प्रदान करते हैं। आप उनके कार्यक्रमों और अभियानों में भाग लेने के लिए उनके साथ पंजीकरण कर सकते हैं।",
    "कॅम्पसवर स्वयंसेवाच्या संधी काय आहेत": "NSS (राष्ट्रीय सेवा योजना) आणि विविध सामाजिक क्लब सामुदायिक सेवेसाठी स्वयंसेवेच्या संधी देतात. तुम्ही त्यांच्या कार्यक्रमांमध्ये आणि मोहिमांमध्ये सहभागी होण्यासाठी त्यांच्याकडे नोंदणी करू शकता.",

    "how to start a social awareness campaign": "To start a social awareness campaign, you need to submit a proposal to the student affairs office. The proposal should include the campaign's objective, target audience, and planned activities.",
    "social awareness campaign kaise shuru karein": "Social awareness campaign shuru karne ke liye, aapko Student Affairs office ko ek proposal submit karna hoga. Proposal me campaign ka objective, target audience aur planned activities shamil honi chahiye.",
    "सामाजिक जागरूकता अभियान कैसे शुरू करें": "सामाजिक जागरूकता अभियान शुरू करने के लिए, आपको छात्र मामलों के कार्यालय में एक प्रस्ताव जमा करना होगा। प्रस्ताव में अभियान का उद्देश्य, लक्षित दर्शक और नियोजित गतिविधियां शामिल होनी चाहिए।",
    "सामाजिक जागृती मोहीम कशी सुरू करावी": "सामाजिक जागृती मोहीम सुरू करण्यासाठी, तुम्हाला विद्यार्थी व्यवहार कार्यालयात एक प्रस्ताव सादर करणे आवश्यक आहे. प्रस्तावात मोहिमेचा उद्देश, लक्ष्यित प्रेक्षक आणि नियोजित क्रियाकलाप यांचा समावेश असावा.",

    # 48 Library Services
    "how to request a book not available in the library": "You can place an inter-library loan request through the library portal or by speaking to a librarian. The library will try to acquire the book from a partner institution.",
    "library me jo book nahi hai usse kaise request karein": "Aap library portal ke through ya librarian se baat karke inter-library loan request kar sakte hain. Library partner institution se book dilane ki koshish karegi.",
    "लाइब्रेरी में जो किताब उपलब्ध नहीं है, उसे कैसे मंगाएं": "आप लाइब्रेरी पोर्टल के माध्यम से या लाइब्रेरियन से बात करके इंटर-लाइब्रेरी लोन अनुरोध कर सकते हैं। लाइब्रेरी किसी पार्टनर संस्थान से किताब प्राप्त करने का प्रयास करेगी।",
    "ग्रंथालयात उपलब्ध नसलेल्या पुस्तकाची मागणी कशी करावी": "तुम्ही ग्रंथालय पोर्टलद्वारे किंवा ग्रंथपालशी बोलून आंतर-ग्रंथालय कर्ज विनंती करू शकता. ग्रंथालय भागीदार संस्थेकडून पुस्तक मिळवण्याचा प्रयत्न करेल.",

    "can I reserve a book online": "Yes, you can reserve a book online through the library's catalog portal. You will receive a notification when the book is available for pickup.",
    "kya main online book reserve kar sakta hoon": "Haan, aap library ke catalog portal ke through online book reserve kar sakte hain. Jab book pickup ke liye available hogi, tab aapko notification milega.",
    "क्या मैं ऑनलाइन किताब आरक्षित कर सकता हूँ": "हाँ, आप लाइब्रेरी के कैटलॉग पोर्टल के माध्यम से ऑनलाइन किताब आरक्षित कर सकते हैं। जब किताब लेने के लिए उपलब्ध होगी, तब आपको एक सूचना मिलेगी।",
    "मी ऑनलाइन पुस्तक आरक्षित करू शकतो का": "हो, तुम्ही ग्रंथालयाच्या कॅटलॉग पोर्टलद्वारे ऑनलाइन पुस्तक आरक्षित करू शकता. पुस्तक घेण्यासाठी उपलब्ध झाल्यावर तुम्हाला सूचना मिळेल.",
    
    # 49 Hostel & Mess
    "what are the hostel visitor rules": "Visitors are only allowed in the common areas of the hostel during specific visiting hours. They must register at the reception with a valid ID.",
    "hostel me visitor rules kya hain": "Visitors ko hostel ke common areas me sirf specific visiting hours me hi allow kiya jata hai. Unhe valid ID ke saath reception par register karna zaroori hai.",
    "छात्रावास में आगंतुक नियम क्या हैं": "आगंतुकों को छात्रावास के सामान्य क्षेत्रों में केवल विशिष्ट मुलाकात के घंटों के दौरान ही अनुमति है। उन्हें एक वैध आईडी के साथ रिसेप्शन पर पंजीकरण करना होगा।",
    "वसतिगृहात पाहुण्यांसाठी काय नियम आहेत": "पाहुण्यांना वसतिगृहाच्या सामान्य भागात केवळ विशिष्ट वेळेतच परवानगी आहे. त्यांना वैध ओळखपत्रासह रिसेप्शनवर नोंदणी करावी लागेल.",

    "are laundry services available in the hostel": "Yes, most hostels have laundry services or a designated laundry area with washing machines. Check with your hostel warden for details and payment options.",
    "kya hostel me laundry services available hain": "Haan, adhiktar hostels me laundry services ya washing machines ke saath ek designated laundry area hota hai. Details aur payment options ke liye apne hostel warden se check karein.",
    "क्या छात्रावास में कपड़े धोने की सेवाएँ उपलब्ध हैं": "हाँ, अधिकांश छात्रावासों में कपड़े धोने की सेवाएँ या वॉशिंग मशीन के साथ एक निर्दिष्ट कपड़े धोने का क्षेत्र होता है। विवरण और भुगतान विकल्पों के लिए अपने छात्रावास वार्डन से जांच करें।",
    "वसतिगृहात कपडे धुण्याची सेवा उपलब्ध आहे का": "हो, बहुतेक वसतिगृहांमध्ये कपडे धुण्याची सेवा किंवा वॉशिंग मशीनसह एक निश्चित कपडे धुण्याचे ठिकाण असते. तपशील आणि पेमेंट पर्यायांसाठी तुमच्या वसतिगृह वॉर्डनशी संपर्क साधा.",

    # 50 Student Forms & Documents
    "how to get a bonafide certificate": "You can request a bonafide certificate through the student portal by filling out a form and paying a nominal fee. The certificate can be collected from the registrar's office after a few days.",
    "bonafide certificate kaise milega": "Aap student portal ke through ek form bhar kar aur thodi fees dekar bonafide certificate request kar sakte hain. Certificate kuch dino baad registrar office se collect kiya ja sakta hai.",
    "बोनाफाइड सर्टिफिकेट कैसे मिलेगा": "आप छात्र पोर्टल के माध्यम से एक फॉर्म भरकर और एक मामूली शुल्क का भुगतान करके बोनाफाइड सर्टिफिकेट का अनुरोध कर सकते हैं। प्रमाण पत्र कुछ दिनों बाद रजिस्ट्रार के कार्यालय से एकत्र किया जा सकता है।",
    "बोनाफाईड प्रमाणपत्र कसे मिळेल": "तुम्ही विद्यार्थी पोर्टलद्वारे एक फॉर्म भरून आणि नाममात्र शुल्क भरून बोनाफाईड प्रमाणपत्र मागू शकता. काही दिवसांनी प्रमाणपत्र रजिस्ट्रारच्या कार्यालयातून गोळा करता येते."
}
SUGGESTIONS = {
    "English":  ["What is your name?", "Good morning", "When does the timetable come?",
                 "What is AI?", "When are exams conducted?", "What is placement process?"],
    "Hindi":    ["आपका नाम क्या है", "सुप्रभात", "समय सारिणी कब आती है",
                 "AI क्या है", "परीक्षाएँ कब होती हैं", "प्लेसमेंट प्रक्रिया क्या है"],
    "Marathi":  ["तुमचं नाव काय आहे", "शुभ प्रभात", "वेळापत्रक कधी येते",
                 "AI काय आहे", "परीक्षा कधी होतात", "प्लेसमेंट प्रक्रिया काय आहे"],
    "Gujarati": ["તમારું નામ શું છે", "સુપ્રભાત", "સમયપત્રક ક્યારે આવે છે",
                 "AI શું છે", "પરીક્ષા ક્યારે લેવાય છે", "પ્લેસમેન્ટ પ્રક્રિયા શું છે"],
    "Bengali":  ["তোমার নাম কি", "সুপ্রভাত", "ক্লাসের রুটিন কবে বের হয়",
                 "AI কি", "পরীক্ষা কবে হয়", "প্লেসমেন্ট প্রক্রিয়া কি"],
}

# ── Precompute embeddings ─────────────────────────────────────────────────────
faq_questions = list(faqs.keys())
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

def chatbot(query: str, lang: str) -> dict:
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, faq_embeddings)[0]
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])

    if best_score > 0.6:
        return {"reply": faqs[faq_questions[best_idx]], "is_fallback": False}
    return {"reply": FALLBACK.get(lang, FALLBACK["English"]), "is_fallback": True}

# ── HTML Template ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>🤖 Multilingual Bot Assistant</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #22263a;
    --accent: #6c63ff;
    --accent2: #a78bfa;
    --user-bubble: #2d2f5e;
    --bot-bubble: #1e2135;
    --text: #e8eaf0;
    --text-muted: #7c82a0;
    --border: #2a2e45;
    --success: #10b981;
    --radius: 16px;
  }

  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* Header */
  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-shrink: 0;
    z-index: 10;
  }
  .bot-avatar {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
  }
  header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: -0.3px;
  }
  header p { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
  .online-dot {
    width: 8px; height: 8px;
    background: var(--success);
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* Language selector */
  .lang-bar {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 10px 24px;
    display: flex;
    gap: 8px;
    flex-shrink: 0;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .lang-bar::-webkit-scrollbar { display: none; }
  .lang-btn {
    padding: 5px 14px;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-muted);
    font-size: 13px;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.2s;
  }
  .lang-btn:hover { border-color: var(--accent); color: var(--accent2); }
  .lang-btn.active {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
    font-weight: 600;
  }

  /* Chat area */
  #chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 24px 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    scroll-behavior: smooth;
  }
  #chat-area::-webkit-scrollbar { width: 4px; }
  #chat-area::-webkit-scrollbar-track { background: transparent; }
  #chat-area::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  .msg-row {
    display: flex;
    gap: 10px;
    max-width: 780px;
    width: 100%;
    animation: fadeUp 0.25s ease;
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .msg-row.user { align-self: flex-end; flex-direction: row-reverse; }
  .msg-row.bot  { align-self: flex-start; }

  .avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
  }
  .avatar.bot-av { background: linear-gradient(135deg, var(--accent), var(--accent2)); }
  .avatar.user-av { background: var(--user-bubble); border: 1px solid var(--border); }

  .bubble {
    padding: 11px 16px;
    border-radius: var(--radius);
    font-size: 14.5px;
    line-height: 1.55;
    max-width: 70%;
    word-break: break-word;
  }
  .user .bubble {
    background: var(--user-bubble);
    border-bottom-right-radius: 4px;
  }
  .bot .bubble {
    background: var(--bot-bubble);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }

  /* Contact buttons inside fallback */
  .contact-btns { display: flex; gap: 8px; margin-top: 10px; }
  .contact-btn {
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 13px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    text-decoration: none;
    transition: opacity 0.2s;
  }
  .contact-btn:hover { opacity: 0.85; }
  .whatsapp-btn { background: #25d366; color: #fff; }
  .call-btn { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }

  /* Suggestions */
  #suggestions-wrap {
    padding: 10px 16px;
    border-top: 1px solid var(--border);
    background: var(--surface);
    flex-shrink: 0;
  }
  #suggestions-wrap p { font-size: 11px; color: var(--text-muted); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.6px; }
  #suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
  }
  .sug-btn {
    padding: 6px 13px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    color: var(--text);
    font-size: 12.5px;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
  }
  .sug-btn:hover { border-color: var(--accent2); color: var(--accent2); }

  /* Input bar */
  .input-bar {
    padding: 14px 16px;
    background: var(--surface);
    border-top: 1px solid var(--border);
    display: flex;
    gap: 10px;
    flex-shrink: 0;
  }
  #user-input {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 11px 16px;
    color: var(--text);
    font-size: 14px;
    font-family: 'Inter', sans-serif;
    outline: none;
    transition: border-color 0.2s;
  }
  #user-input:focus { border-color: var(--accent); }
  #user-input::placeholder { color: var(--text-muted); }
  #send-btn {
    width: 44px; height: 44px;
    background: var(--accent);
    border: none;
    border-radius: 12px;
    color: #fff;
    font-size: 18px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.2s, transform 0.1s;
    flex-shrink: 0;
  }
  #send-btn:hover { background: var(--accent2); }
  #send-btn:active { transform: scale(0.93); }

  /* Typing indicator */
  .typing { display: flex; gap: 4px; align-items: center; padding: 4px 0; }
  .typing span {
    width: 7px; height: 7px;
    background: var(--text-muted);
    border-radius: 50%;
    animation: bounce 1.2s infinite;
  }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-6px); }
  }

  /* Welcome message */
  .welcome {
    text-align: center;
    padding: 32px 16px;
    color: var(--text-muted);
  }
  .welcome .big-emoji { font-size: 48px; margin-bottom: 12px; }
  .welcome h2 { font-family: 'Space Grotesk', sans-serif; font-size: 18px; color: var(--text); margin-bottom: 6px; }
  .welcome p { font-size: 13.5px; }
</style>
</head>
<body>

<header>
  <div class="bot-avatar">🤖</div>
  <div>
    <h1>Multilingual Bot Assistant</h1>
    <p><span class="online-dot"></span>Online · Responds instantly</p>
  </div>
</header>

<div class="lang-bar" id="lang-bar">
  <button class="lang-btn active" onclick="setLang('English', this)">🇬🇧 English</button>
  <button class="lang-btn" onclick="setLang('Hindi', this)">🇮🇳 हिन्दी</button>
  <button class="lang-btn" onclick="setLang('Marathi', this)">🇮🇳 मराठी</button>
  <button class="lang-btn" onclick="setLang('Gujarati', this)">🇮🇳 ગુજરાતી</button>
  <button class="lang-btn" onclick="setLang('Bengali', this)">🇧🇩 বাংলা</button>
</div>

<div id="chat-area">
  <div class="welcome">
    <div class="big-emoji">👋</div>
    <h2>Hello! How can I help you today?</h2>
    <p>Ask me anything or tap a suggestion below.</p>
  </div>
</div>

<div id="suggestions-wrap">
  <p>💡 Try asking</p>
  <div id="suggestions"></div>
</div>

<div class="input-bar">
  <input type="text" id="user-input" placeholder="Type your question here…" autocomplete="off"/>
  <button id="send-btn" onclick="sendMessage()">➤</button>
</div>

<script>
const SUGGESTIONS = {{ suggestions|tojson }};
let currentLang = 'English';

function setLang(lang, btn) {
  currentLang = lang;
  document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderSuggestions();
}

function renderSuggestions() {
  const wrap = document.getElementById('suggestions');
  wrap.innerHTML = '';
  (SUGGESTIONS[currentLang] || []).forEach(s => {
    const b = document.createElement('button');
    b.className = 'sug-btn';
    b.textContent = s;
    b.onclick = () => sendMessage(s);
    wrap.appendChild(b);
  });
}

function appendMsg(speaker, text, isFallback=false) {
  const area = document.getElementById('chat-area');
  const welcome = area.querySelector('.welcome');
  if (welcome) welcome.remove();

  const row = document.createElement('div');
  row.className = `msg-row ${speaker}`;

  const av = document.createElement('div');
  av.className = `avatar ${speaker === 'user' ? 'user-av' : 'bot-av'}`;
  av.textContent = speaker === 'user' ? '🧑' : '🤖';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  if (isFallback) {
    const contactWrap = document.createElement('div');
    contactWrap.className = 'contact-btns';
    contactWrap.innerHTML = `
      <a class="contact-btn whatsapp-btn" href="https://wa.me/917066853631" target="_blank">💬 WhatsApp</a>
      <a class="contact-btn call-btn" href="tel:7066853631">📞 Call</a>
    `;
    bubble.appendChild(contactWrap);
  }

  row.appendChild(av);
  row.appendChild(bubble);
  area.appendChild(row);
  area.scrollTop = area.scrollHeight;
}

function showTyping() {
  const area = document.getElementById('chat-area');
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  row.id = 'typing-row';

  const av = document.createElement('div');
  av.className = 'avatar bot-av';
  av.textContent = '🤖';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';

  row.appendChild(av);
  row.appendChild(bubble);
  area.appendChild(row);
  area.scrollTop = area.scrollHeight;
}

function removeTyping() {
  const t = document.getElementById('typing-row');
  if (t) t.remove();
}

async function sendMessage(text) {
  const input = document.getElementById('user-input');
  const query = text || input.value.trim();
  if (!query) return;

  input.value = '';
  appendMsg('user', query);
  showTyping();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, lang: currentLang })
    });
    const data = await res.json();
    removeTyping();
    appendMsg('bot', data.reply, data.is_fallback);
  } catch (e) {
    removeTyping();
    appendMsg('bot', '⚠️ Something went wrong. Please try again.');
  }
}

document.getElementById('user-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') sendMessage();
});

// Init suggestions
renderSuggestions();
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML, suggestions=SUGGESTIONS)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query', '').strip()
    lang  = data.get('lang', 'English')
    if not query:
        return jsonify({"reply": "Please type something!", "is_fallback": False})
    result = chatbot(query, lang)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
