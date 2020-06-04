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
const dbName string = "backend-app"
const devEnv = false

// insertQuery holds default string to insert item into database
var insertQuery = fmt.Sprintf("INSERT INTO %s VALUES (?,?,?)", dbName)

var db *sql.DB
var connected bool

// ErrNotConnected should be thrown when there is no established connection to the database
var ErrNotConnected error = errors.New("No connection to database")

// database contains the general table to store username and given score for the item
type database struct {
	userName  string
	userScore float64
	imgPath   string
}

// CO2 contains data about item with agrigation sum of greenhouse gasses (CO2, NH4, etc)
type CO2 struct {
	items    string
	emission float64
}

// connectDB allows to connect to database
func connectDB(database, pwd string) *sql.DB {
	var cnnstr string
	if devEnv {
		cnnstr = fmt.Sprintf("root@tcp(localhost)/%s", dbName)
	} else {
		cnnstr = fmt.Sprintf("%s:%s@tcp(localhost)/%s", database, pwd, dbName)
	}
	log.Info("try to connect db with ", cnnstr)
	db, _ := sql.Open("mysql", cnnstr)
	return db
}

func createTable(table string) error {
	connected = true
	// added connection string
	_, err := db.Exec("CREATE TABLE IF NOT EXISTS %s (userName varchar(255), userScore int, imgPath varchar(255));")
	return err
}

func init() {
	connected = false
	db = connectDB()
}

func getUser(name string) error {
	return nil
}
