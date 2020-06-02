package db

import (
	"database/sql"
	"errors"
	"fmt"

	log "github.com/sirupsen/logrus"

	// mysql driver
	_ "github.com/go-sql-driver/mysql"
)

//constant
const dbName string = "backend_db"
const insertQuery string = fmt.Sprintf("INSERT INTO %s VALUES (?,?,?)", dbName)

var db *sql.DB
var connected bool

// ErrNotConnected should be thrown when there is no established connection to the database
var ErrNotConnected error = errors.New("No connection to database")

const devEnv = false

// User contains user-related info
type User struct {
	name   string
	score  int64
	images string
}

// CO2 contains co2-related data
type CO2 struct {
	items   string
	details float64
}

// ImgPath contains string for images on storage bucket
type ImgPath struct {
	path string
}

// database : type Database {slices of multiple struct}
type database struct {
	user   User
	co2    CO2
	impath ImgPath
}

// connectDB allows to connect to database
func connectDB() *sql.DB {
	var cnnstr string
	if devEnv {
		cnnstr = fmt.Sprintf("root@tcp(localhost)/%s", dbName)
	} else {
		//TODO: added cnnstr when not in local deployment
		cnnstr = fmt.Sprintf("root@tcp(localhost)/%s", dbName)
	}
	log.Info("try to connect db with ", cnnstr)
	db, _ := sql.Open("mysql", cnnstr)
	return db
}

func createTable() error {
	connected = true
	// added connection string
	_, err := db.Exec("CREATE TABLE IF NOT EXISTS %s ()")
	return err
}

func init() {
	connected = false
	db = connectDB()
}

func getUser(name string) error {
	return nil
}
