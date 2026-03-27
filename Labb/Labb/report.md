# Rapport: Filmrekommendationssystem med TF-IDF

## Introduktion / Problemställning
Syftet med denna studie var att utveckla ett rekommendationssystem för filmer baserat på innehåll. Problemet bestod i att, givet en film, identifiera liknande filmer utifrån deras egenskaper såsom genre, titel och användargenererade taggar.

Rekommendationssystem är en central del inom maskininlärning och används för att filtrera och föreslå relevant innehåll. I detta arbete användes en innehållsbaserad metod där likheten mellan filmer beräknas utifrån textrepresentationer.

En central metod är TF-IDF (Term Frequency–Inverse Document Frequency), som "viktar" ord baserat på hur viktiga de är i ett dokument relativt hela datamängden. 

Likheten mellan filmer beräknas sedan med cosinuslikhet:

\[
\cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}
\]

där \( A \) och \( B \) representerar filmers TF-IDF-vektorer.

## Data-analys (EDA)
Datamängden bestod av två filer: `movies.csv` och `tags.csv`. Filmerna innehöll information om titel och genre, medan taggarna representerade användargenererad metadata.

För att möjliggöra analys transformerades datan till en textbaserad representation. Texten normaliserades genom konvertering till gemener och borttagning av specialtecken. Genrer omvandlades till tokeniserad text och taggar aggregerades per film.

Analysen visade följande relevanta samband:

- **Genre och likhet:** Filmer med samma eller överlappande genrer tenderar att ha hög cosinuslikhet, eftersom dessa ord förekommer frekvent i samma dokument. Genre fungerar därmed som en stark grundsignal för likhet.

- **Taggars betydelse:** Taggar bidrar med mer specifik semantisk information än genrer. Exempelvis kan två filmer dela genre men särskiljas genom taggar som beskriver tema, ton eller stil. Detta förbättrar modellens förmåga att särskilja annars liknande filmer.

- **Textrepresentationens påverkan:** Kombinationen av flera textkällor (`genres`, `title`, `tags`) ger en bättre representation än enskilda komponenter. Särskilt taggar i kombination med genrer ger en mer nyanserad likhetsbedömning.

- **N-gram-effekt:** Användningen av bigrams (tvåords-kombinationer) gör att modellen kan fånga kontext, exempelvis uttryck som “science fiction” eller “romantic comedy”, vilket förbättrar precisionen jämfört med enbart enskilda ord.

- **Datans begränsningar:** Vissa filmer saknar taggar, vilket innebär att deras representation i högre grad baseras på genrer och titel. Detta kan leda till mindre precisa rekommendationer för dessa filmer.

Sammanfattningsvis visar analysen att både genre och taggar har stor påverkan på modellens prestanda, där genrer ger en grov gruppering medan taggar möjliggör finare semantisk differentiering.

## Modell
Flera metoder för rekommendation finns, men i detta arbete användes en innehållsbaserad modell med TF-IDF-vektorisering.

TF-IDF-vektoriseraren konfigurerades med följande parametrar:
- stop words: engelska stoppord togs bort
- n-gram: (1,2) vilket innebär både enstaka ord och tvåords-kombinationer
- max features: 200000

Efter vektorisering representerades varje film som en högdimensionell vektor.

Likheten mellan filmer beräknades med cosinuslikhet och de k mest lika filmerna returnerades. För att undvika att en film rekommenderar sig själv exkluderades den från resultatet.

## Resultat
Modellen testades genom att ange en film, exempelvis "Toy Story". Systemet returnerade de k mest liknande filmerna baserat på textlikhet.

Resultaten visade att:
- Filmer med liknande genre och teman prioriterades
- Taggar bidrog till mer nyanserade rekommendationer
- Titeln hade mindre påverkan jämfört med genre och taggar

Exempel på output: ("Toy Story")

| movieId | title        | genres      | similarity |
|-----|--------------|-------------|--------|
|3114|Toy Story 2 (1999)| Animation.. | 0.838732|
|2355|Bug's Life, A (1998)| Animation.. |0.715483|
|78499|Toy Story 3 (2010)| Animation.. |0.633648|
|4886|Monsters, Inc. (2001)| Animation.. |0.593836|

Modellen gav relevanta rekommendationer och visade god förmåga att identifiera liknande innehåll.

## Diskussion
Resultaten visar att en TF-IDF-baserad modell fungerar väl för innehållsbaserade rekommendationer. Metoden är relativt enkel att implementera och kräver ingen användarhistorik.

Begränsningar inkluderar:
- Ingen hänsyn tas till användarbeteende
- Rekommendationerna baseras enbart på textlikhet
- Kvaliteten beror på hur informativa taggarna är

Möjliga vidareutvecklingar innefattar:
- Kollaborativ filtrering
- Hybridmodeller
- Viktning av olika textkomponenter

Sammanfattningsvis visar modellen att textbaserad representation av filmer är tillräcklig för att generera relevanta rekommendationer, men att mer avancerade metoder kan förbättra systemets precision.