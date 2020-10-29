import numpy as np
import numpy.core.defchararray as npc
from random import randint
import matplotlib.pyplot as plt

"""
master data makanan, berupa list dua dimensi, dimensi pertama adalah jenis makanan yaitu buah,
karbohidrat dan lauk. Kemudian dimensi kedua adalah terdiri dari binary sebagai indetifikasi gen,
nama makanan dan jumlah kalori
"""
data_makanan = np.array([
    [["00", "Apel", 52], ["01", "Pisang", 89], ["10", "Mangga", 60], ["11", "Semangka", 30]],
    [["00", "Roti", 265], ["01", "Kentang", 87], ["10", "Jagung", 86], ["11", "Ketela", 159]],
    [["00", "Ikan", 206], ["01", "Daging Sapi", 143], ["10", "Ayam", 239], ["11", "Telur", 155]]
])

# maks_kalori = int(500);
# probabilitas_kawin = int(70)
"""
Membuat satu kombinasi makanan, satu kombinasi terdiri dari satu buah, satu karbohidrat dan
satu lauk. kombinasi bersifat random.
kombinasi satu kali makan mewakili satu kromosom
"""
def menu_sekali_makan():
    kromosom = np.array([])
    for menu in data_makanan :
        random_index    = randint(0, 3)
        kromosom = np.append(kromosom, menu[random_index, 0])
    return kromosom

"""
Satu individu diwujudkan dalam menu seminggu. menu seminggu inilah yang akan dihitung fitness 
nya terhadap kriteria fitness.
"""
def menu_seminggu():
    individu = np.array([])
    for hari in range(7) :
        individu = np.append(individu, menu_sekali_makan())
    individu = individu.reshape(7,3)
    return individu

"""
Membangkitkan generasi pertama
"""
def buat_leluhur(jumlah_populasi):
    leluhur = np.array([])
    for i in range(jumlah_populasi):
        leluhur = np.append(leluhur, menu_seminggu())
    leluhur = leluhur.reshape(jumlah_populasi, 7, 3)
    # print(leluhur)
    return leluhur

"""
Cek fitness untuk setiap generasi yang dilahirkan.
cek fitness I adalah menghitung total kalori untuk setiap kali makan. jika > 500 kalori maka
individu tersebut dinyatakan kurang fit. jika kurang dari 500 maka diberikan skor fitnes 10
"""
def dibawah_kalori_maksimal(menu_sekali_makan, maks_kalori):
    # cari kalori buah
    buah, cols = np.where(data_makanan[0] == menu_sekali_makan[0])
    kalori_buah = int(data_makanan[0][buah, 2][0])
    
    # cari kalori karbo
    karbo, cols = np.where(data_makanan[1] == menu_sekali_makan[1])
    kalori_karbo = int(data_makanan[1][karbo, 2][0])

    # cari kalori lauk
    lauk, cols = np.where(data_makanan[2] == menu_sekali_makan[2])
    kalori_lauk = int(data_makanan[2][lauk, 2][0])
    
    # jumlahkan total kalori
    total_kalori = kalori_buah + kalori_karbo + kalori_lauk
    # print(total_kalori)
    # cek apakah kalori melebihi batas maksimal
    if total_kalori < maks_kalori :
        return True
    else:
        return False

"""
Cek menu sebelumnya, skor diberikan jika menu tidak sama, minimal skor 0 maksimal 15.
"""
def cek_menu_sebelumnya(menu_sebelum, menu_sekarang) :
    skor = 0
    if menu_sebelum[0] != menu_sekarang[0] :
        skor += 5
    if menu_sebelum[1] != menu_sekarang[1] :
        skor += 5
    if menu_sebelum[2] != menu_sekarang[2] :
        skor += 5
    return skor

"""
cek fitness setiap individu dalam satu generasi
"""
def cek_fitness(generasi, maks_kalori):
    fit_status_individu = np.array([], dtype='int16')
    nomor_individu = 0
    best_score  =   0
    worst_score =   1000
    best_individu = np.array([])
    worst_individu = np.array([])
    for individu in generasi :
        fit_status_individu = np.append(fit_status_individu, nomor_individu)
        nomor_individu += 1
        skor = 0
        kalorifit = 60
        menu_sebelum = None
        for kromosom in individu :
            if dibawah_kalori_maksimal(kromosom, maks_kalori) == False :
                kalorifit = 0
            if menu_sebelum is not None :
                skor = skor + cek_menu_sebelumnya(menu_sebelum, kromosom)
            menu_sebelum = kromosom
        fit_status_individu = np.append(fit_status_individu, kalorifit)
        fit_status_individu = np.append(fit_status_individu, skor)
        fit_status_individu = np.append(fit_status_individu, kalorifit + skor)
        if (kalorifit + skor) > best_score:
            best_score  =   kalorifit + skor
            best_individu = individu
        if worst_score > (kalorifit + skor):
            worst_score =   kalorifit + skor
            worst_individu = individu
    fit_status_individu = fit_status_individu.reshape(nomor_individu, 4)
    # print(fit_status_individu)
    fit_stat = fit_status_individu[:, [3]]
    # cari yang terbaik berdasarkan skor terbaik
    # get best score
    # best = np.amax(fit_stat)
    # worst= np.amin(fit_stat)
    # print(fit_stat)
    # print("nilai individu terbaik = " + str(np.amax(fit_stat)) + " Nilai individu terburuk = "+ str(np.amin(fit_stat)))
    return fit_status_individu, best_score, worst_score, best_individu, worst_individu

"""
membuat persentase kemungkinan berpasangan dengan roulete.
"""          
def make_roulete (fit_status_individu):
    sum_fit_status_individu = fit_status_individu.sum(axis=0)
    new_fit_status_individu = np.array([], dtype='int16')
    angka = 0
    for individu in fit_status_individu:
        angka = angka + int(round(individu[3]/sum_fit_status_individu[3]*100,0))
        individu = np.append(individu, angka)
        new_fit_status_individu = np.append(new_fit_status_individu, individu)
    new_fit_status_individu = new_fit_status_individu.reshape(len(fit_status_individu), 5)
    # print(new_fit_status_individu)
    return new_fit_status_individu

# cari yang beruntung untuk kawin dan carikan pasangannya
def do_roulete(fit_status_individu, jumlah_populasi):
    jumlah_pasangan = int(jumlah_populasi/2) #untuk buat generasi dengan ukuran populasi sama dibutuhkan pasangan sejumlah 1/2 jumlah populasi
    max_fit = np.amax(fit_status_individu[:,[4]]) # untuk mencari jumlah persentase
    # print(max_fit)
    pasangan = np.array([], dtype='int16')
    # print(fit_status_individu)  
    # membuat pasangan sejumlah yang ditetapkan
    individu_sebelumnya = None
    for pop in range(jumlah_populasi):
        random_index = randint(1, max_fit) #nomer keberuntungan
        #ambil satu individu yang beruntung
        for individu in fit_status_individu:
            if random_index <= individu[4]:
                individu_sebelumnya = individu[0]
                pasangan = np.append(pasangan, individu[0])
                break
    # print(fit_status_individu)
    # print(pasangan)
    pasangan = pasangan.reshape(int(len(pasangan)/2), 2) #reshape sesuai panjang array dibagi 2
    return pasangan

def mutasi (individu):
    for i in range(2):
        random_index = randint(0, 6)
        ori     = individu[random_index]
        mutan   = npc.replace(ori, "0", "2")
        mutan   = npc.replace(mutan, "1", "0")
        mutan   = npc.replace(mutan, "2", "1")
        individu[random_index] = mutan
    return individu

def kawin (generasi, pasangan, probabilitas_kawin, probabilitas_mutasi) :
    keturunan = np.array([])
    # print(pasangan)
    for couple in pasangan:
        # print(generasi[couple[1]])
        parent1 = generasi[couple[0]]
        parent2 = generasi[couple[1]]
        child1  = parent1
        child2  = parent2
        # print("=====")
        # print(child2)
        random_chance = randint(0, 100)
        # proses kawin
        if random_chance > (100 - probabilitas_kawin) :
            for i in range(3) :
                random_index = randint(0, 6)
                child1[random_index] = parent2[random_index] 
            for i in range(3) :
                random_index = randint(0, 6)
                child2[random_index] = parent1[random_index] 
                # print("=====")
                # print(child2[random_index])

        # proses probabilitas mutasi generasi baru
        # print(child2)
        # print("=====")
        random_mutate   = randint(0, probabilitas_mutasi)
        random_match    = randint(0, probabilitas_mutasi)
        if random_mutate == random_match:
            child2 = mutasi(child2)
            child1 = mutasi(child1)

        keturunan = np.append(keturunan, child1)
        keturunan = np.append(keturunan, child2)
    # print(keturunan)
    keturunan = keturunan.reshape(-1, 7, 3)
    # print(generasi1)
    return keturunan

def eksekusi(jml_populasi, jml_iterasi, maks_kalori, probabilitas_kawin, probabilitas_mutasi):
    # print(type(maks_kalori))
    bw_individu   = np.array([], dtype='int16')
    it              = int(0)
    generasi1       = buat_leluhur(jml_populasi)    
    fit_status      = cek_fitness(generasi1, maks_kalori)
    best_score_of_all  = fit_status[1]
    worst_score_of_all = fit_status[2]
    best_of_all     = fit_status[3]
    worst_of_all    = fit_status[4]
    bw_individu     = np.append(bw_individu, it)
    bw_individu     = np.append(bw_individu, int(fit_status[1]))
    bw_individu     = np.append(bw_individu, int(fit_status[2]))
    new_fit_status  = (make_roulete(fit_status[0]))
    pasangan        = do_roulete(new_fit_status, jml_populasi)
    keturunan       = kawin(generasi1, pasangan, probabilitas_kawin, probabilitas_mutasi)
    for it in range(jml_iterasi):
        fit_status      = cek_fitness(keturunan, maks_kalori)
        if fit_status[1] > best_score_of_all:
            best_score_of_all = fit_status[1]
            best_of_all = fit_status[3]
        if worst_score_of_all > fit_status[2]:
            worst_score_of_all = fit_status[2]
            worst_of_all = fit_status[4]
        bw_individu     = np.append(bw_individu, it+1)
        bw_individu     = np.append(bw_individu, int(fit_status[1]))
        bw_individu     = np.append(bw_individu, int(fit_status[2]))
        new_fit_status  = (make_roulete(fit_status[0]))
        pasangan        = do_roulete(new_fit_status, jml_populasi)
        keturunan       = kawin(keturunan, pasangan, probabilitas_kawin, probabilitas_mutasi)
    bw_individu = bw_individu.reshape(jml_iterasi+1, 3)
    # print(bw_individu)
    return bw_individu, best_of_all, worst_of_all

def printhasil(best, worst):
    hari        = 1
    print("Susunan Menu terbaik : ")
    for menu in best:
        idxbuah, cols   = np.where(data_makanan[0] == menu)
        nama_buah       = data_makanan[0][idxbuah, 1][0]
        idxkarbo, cols   = np.where(data_makanan[1] == menu)
        nama_karbo       = data_makanan[1][idxkarbo, 1][0]
        idxlauk, cols   = np.where(data_makanan[2] == menu)
        nama_lauk       = data_makanan[2][idxlauk, 1][0]
        print("Hari ke-"+str(hari)+", Buah :"+nama_buah+", makanan : "+nama_karbo+", Lauk: "+nama_lauk)
        hari += 1
    print("========================================================================")    
    print("")    
    print("Susunan Menu Terburuk : ")    
    hari = 1
    for menu in worst:
        idxbuah, cols   = np.where(data_makanan[0] == menu)
        nama_buah       = data_makanan[0][idxbuah, 1][0]
        idxkarbo, cols   = np.where(data_makanan[1] == menu)
        nama_karbo       = data_makanan[1][idxkarbo, 1][0]
        idxlauk, cols   = np.where(data_makanan[2] == menu)
        nama_lauk       = data_makanan[2][idxlauk, 1][0]
        print("Hari ke-"+str(hari)+", Buah :"+nama_buah+", makanan : "+nama_karbo+", Lauk: "+nama_lauk)
        hari += 1
    return hari

print("=======================================================")
print("=============Penerapan Algoritma Genetika==============")
print("======================== Pada =========================")
print("=================Penyusunan Menu Makan=================")
print("=======================================================")
print("")
print("Masukkan Jumlah Populasi :")
populasi = int(input())
print("Masukkan Jumlah Iterasi / Generasi :")
iterasi = int(input())
print("")
print("=================*********************=================")
print("=================********Hasil********=================")
print("")
hasil = eksekusi(populasi, iterasi, 400, 70, 100)
print("Binary Individu terbaik")
print(hasil[1])
print("")
print("Binary Individu terburuk")
print(hasil[2])
print("")
print("=================*********************=================")
printhasil(hasil[1], hasil[2])
bw_individu     = hasil[0]
generation      = bw_individu[:, [0]]
best            = bw_individu[:, [1]]
worst           = bw_individu[:, [2]]
generation      = generation.reshape(-1)
best            = best.reshape(-1)
worst           = worst.reshape(-1)


"""
Plot Hasil
"""
ypoints = np.array(best)
ypoints2 = np.array(worst)

plt.title("Perkembangan Terbaik dan Terburuk Pada Generasi")
plt.plot(ypoints, label="Terbaik")
plt.plot(ypoints2, label="terburuk")
plt.xlabel("Generasi ke")
plt.ylabel("Skor")
plt.legend()
plt.show()