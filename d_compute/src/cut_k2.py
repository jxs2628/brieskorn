import csv 
 
with open("brieskorn_d_all_800.csv","r") as source:
    rdr= csv.reader( source )
    with open("brieskorn_800_only_d.csv","w") as target:
        wtr= csv.writer( target )
        for r in rdr:
            wtr.writerow( (r[0], r[1], r[2], r[3]) )
